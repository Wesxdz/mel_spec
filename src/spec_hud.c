#define _GNU_SOURCE  // For strcasestr
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strings.h>  // For strcasecmp
#include <sndfile.h>
#include <fftw3.h>
#include <soundio/soundio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>  // For clock_gettime
#include <sys/stat.h>
#include <sys/types.h>
#include "fpng/src/fpng.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define N_FFT 1024              // Reduced for lower latency
#define HOP_LENGTH 256          // Reduced proportionally
#define N_MELS 128
#define SR 22050
#define BUFFER_DURATION 2.0f    // Keep 2 seconds of audio in circular buffer
#define WINDOW_DURATION 0.1f    // Process 100ms windows for real-time feel
#define ROLLING_WINDOW_DURATION 5.0f  // 5-second rolling window for output
#define OUTPUT_DIR "spectrograms"  // Directory for output PNG files

// Circular buffer for streaming audio
typedef struct {
    float *data;
    int size;
    int write_pos;
    int read_pos;
    pthread_mutex_t mutex;
    int samples_available;
} CircularBuffer;

// Rolling buffer for mel spectrogram frames
typedef struct {
    float **frames;           // 2D array: [frame_index][mel_bin]
    int max_frames;          // Maximum number of frames to store
    int current_frames;      // Current number of frames stored
    int write_pos;           // Circular buffer write position
} MelRollingBuffer;

// OpenGL context for FBO rendering
typedef struct {
    GLFWwindow* window;
    GLuint fbo;
    GLuint texture;
    GLuint vao;
    GLuint vbo;
    GLuint shader_program;
    int tex_width;
    int tex_height;
    pthread_mutex_t mutex;
    unsigned char* pending_image;  // Image data pending upload
    unsigned char* upload_buffer;  // Stable buffer for GL upload
    int upload_buffer_size;        // Current size of upload buffer
    int pending_width;
    int pending_height;
    int has_pending_update;
} GLContext;

// Global state for streaming mode
typedef struct {
    CircularBuffer buffer;
    int sample_rate;
    int running;
    pthread_t processing_thread;
    const char *output_file;
    int frame_counter;
    MelRollingBuffer mel_buffer;  // Rolling buffer for mel frames
    int audio_read_pos;           // Track position in audio buffer for incremental processing
    GLContext gl_context;         // OpenGL context and FBO
    int use_system_audio;         // Flag to indicate if system audio is being used
    float *audio_conversion_buffer;  // Pre-allocated buffer for audio callback (avoid malloc in RT context)
    int audio_conversion_buffer_size;  // Size of the conversion buffer
} StreamState;


static StreamState g_stream_state = {0};

// Mel scale conversion functions
float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create mel filterbank - matching librosa's implementation
void create_mel_filterbank(float **filterbank, int n_mels, int n_fft, int sr) {
    float fmin = 0.0f;
    float fmax = sr / 2.0f;

    // Create mel points
    float min_mel = hz_to_mel(fmin);
    float max_mel = hz_to_mel(fmax);

    float *mels = (float*)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        mels[i] = min_mel + (max_mel - min_mel) * i / (n_mels + 1);
    }

    // Convert back to Hz
    float *freqs = (float*)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        freqs[i] = mel_to_hz(mels[i]);
    }

    // FFT bin frequencies
    int n_freqs = n_fft / 2 + 1;
    float *fft_freqs = (float*)malloc(n_freqs * sizeof(float));
    for (int i = 0; i < n_freqs; i++) {
        fft_freqs[i] = (float)i * sr / n_fft;
    }

    // Build triangular filters with normalized weights
    for (int m = 0; m < n_mels; m++) {
        float left = freqs[m];
        float center = freqs[m + 1];
        float right = freqs[m + 2];

        for (int k = 0; k < n_freqs; k++) {
            float freq = fft_freqs[k];

            if (freq >= left && freq <= center) {
                filterbank[m][k] = (freq - left) / (center - left);
            } else if (freq > center && freq <= right) {
                filterbank[m][k] = (right - freq) / (right - center);
            } else {
                filterbank[m][k] = 0.0f;
            }
        }

        // Normalize each filter by its sum (matching librosa's norm='slaney')
        float sum = 0.0f;
        for (int k = 0; k < n_freqs; k++) {
            sum += filterbank[m][k];
        }
        if (sum > 0.0f) {
            float norm = 2.0f / (freqs[m + 2] - freqs[m]);
            for (int k = 0; k < n_freqs; k++) {
                filterbank[m][k] *= norm;
            }
        }
    }

    free(mels);
    free(freqs);
    free(fft_freqs);
}

// Apply Hanning window - matching librosa's window
void apply_hanning_window(float *frame, int n_fft) {
    for (int i = 0; i < n_fft; i++) {
        float window = 0.5f - 0.5f * cosf(2.0f * M_PI * i / n_fft);
        frame[i] *= window;
    }
}

// Hot colormap - matching matplotlib's implementation exactly
void apply_hot_colormap(float value, unsigned char *rgb) {
    value = fminf(1.0f, fmaxf(0.0f, value));

    // Matplotlib hot colormap exact implementation
    if (value < 0.365079f) {
        // Dark red to red
        rgb[0] = (unsigned char)(255.0f * (0.0416f + value * 2.7034f));
        rgb[1] = 0;
        rgb[2] = 0;
    } else if (value < 0.746032f) {
        // Red to yellow
        rgb[0] = 255;
        rgb[1] = (unsigned char)(255.0f * (value - 0.365079f) / 0.380953f);
        rgb[2] = 0;
    } else {
        // Yellow to white
        rgb[0] = 255;
        rgb[1] = 255;
        rgb[2] = (unsigned char)(255.0f * (value - 0.746032f) / 0.253968f);
    }
}

// Blue colormap for system audio
void apply_blue_colormap(float value, unsigned char *rgb) {
    value = fminf(1.0f, fmaxf(0.0f, value));

    // Dark blue to bright blue gradient
    if (value < 0.5f) {
        // Dark blue to medium blue
        rgb[0] = 0;
        rgb[1] = (unsigned char)(255.0f * value * 0.4f);
        rgb[2] = (unsigned char)(255.0f * (0.1f + value * 1.8f));
    } else {
        // Medium blue to cyan/light blue
        rgb[0] = (unsigned char)(255.0f * (value - 0.5f) * 1.2f);
        rgb[1] = (unsigned char)(255.0f * (0.2f + (value - 0.5f) * 1.6f));
        rgb[2] = 255;
    }
}

// Load and resample audio - matching librosa.load behavior
float *load_audio(const char *filename, sf_count_t *num_frames, int target_sr) {
    SF_INFO sf_info;
    SNDFILE *sf = sf_open(filename, SFM_READ, &sf_info);

    if (!sf) {
        fprintf(stderr, "Error opening audio file: %s\n", sf_strerror(NULL));
        return NULL;
    }

    // Read all frames
    sf_count_t frames = sf_info.frames;
    float *audio_data = (float*)malloc(frames * sf_info.channels * sizeof(float));
    sf_readf_float(sf, audio_data, frames);
    sf_close(sf);

    // Convert to mono if needed (average channels)
    float *mono_data = (float*)malloc(frames * sizeof(float));
    if (sf_info.channels == 1) {
        memcpy(mono_data, audio_data, frames * sizeof(float));
    } else {
        for (sf_count_t i = 0; i < frames; i++) {
            mono_data[i] = 0.0f;
            for (int ch = 0; ch < sf_info.channels; ch++) {
                mono_data[i] += audio_data[i * sf_info.channels + ch];
            }
            mono_data[i] /= sf_info.channels;
        }
    }
    free(audio_data);

    // Normalize audio to [-1, 1] range (matching librosa)
    float max_val = 0.0f;
    for (sf_count_t i = 0; i < frames; i++) {
        float abs_val = fabsf(mono_data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    if (max_val > 0.0f && max_val > 1.0f) {
        for (sf_count_t i = 0; i < frames; i++) {
            mono_data[i] /= max_val;
        }
    }

    // Resample if necessary
    if (sf_info.samplerate != target_sr) {
        float resample_ratio = (float)target_sr / sf_info.samplerate;
        sf_count_t resampled_frames = (sf_count_t)(frames * resample_ratio);
        float *resampled_data = (float*)malloc(resampled_frames * sizeof(float));

        for (sf_count_t i = 0; i < resampled_frames; i++) {
            float src_idx = i / resample_ratio;
            sf_count_t idx0 = (sf_count_t)floor(src_idx);
            sf_count_t idx1 = idx0 + 1;

            if (idx1 >= frames) {
                resampled_data[i] = mono_data[frames - 1];
            } else {
                float frac = src_idx - idx0;
                resampled_data[i] = mono_data[idx0] * (1.0f - frac) + mono_data[idx1] * frac;
            }
        }

        free(mono_data);
        *num_frames = resampled_frames;
        return resampled_data;
    } else {
        *num_frames = frames;
        return mono_data;
    }
}

// Rolling buffer functions for mel spectrogram frames
void mel_rolling_buffer_init(MelRollingBuffer *mrb, int max_frames, int n_mels) {
    mrb->max_frames = max_frames;
    mrb->current_frames = 0;
    mrb->write_pos = 0;
    mrb->frames = (float**)malloc(max_frames * sizeof(float*));
    for (int i = 0; i < max_frames; i++) {
        mrb->frames[i] = (float*)calloc(n_mels, sizeof(float));
    }
}

void mel_rolling_buffer_free(MelRollingBuffer *mrb) {
    for (int i = 0; i < mrb->max_frames; i++) {
        free(mrb->frames[i]);
    }
    free(mrb->frames);
}

void mel_rolling_buffer_add_frame(MelRollingBuffer *mrb, const float *mel_frame, int n_mels) {
    memcpy(mrb->frames[mrb->write_pos], mel_frame, n_mels * sizeof(float));
    mrb->write_pos = (mrb->write_pos + 1) % mrb->max_frames;
    if (mrb->current_frames < mrb->max_frames) {
        mrb->current_frames++;
    }
}

// Circular buffer functions
void circular_buffer_init(CircularBuffer *cb, int size) {
    cb->data = (float*)calloc(size, sizeof(float));
    cb->size = size;
    cb->write_pos = 0;
    cb->read_pos = 0;
    cb->samples_available = 0;
    pthread_mutex_init(&cb->mutex, NULL);
}

void circular_buffer_free(CircularBuffer *cb) {
    free(cb->data);
    pthread_mutex_destroy(&cb->mutex);
}

void circular_buffer_write(CircularBuffer *cb, const float *data, int count) {
    pthread_mutex_lock(&cb->mutex);

    for (int i = 0; i < count; i++) {
        cb->data[cb->write_pos] = data[i];
        cb->write_pos = (cb->write_pos + 1) % cb->size;

        if (cb->samples_available < cb->size) {
            cb->samples_available++;
        } else {
            // Buffer full, move read position
            cb->read_pos = (cb->read_pos + 1) % cb->size;
        }
    }

    pthread_mutex_unlock(&cb->mutex);
}

int circular_buffer_read(CircularBuffer *cb, float *data, int count) {
    pthread_mutex_lock(&cb->mutex);

    int to_read = (count < cb->samples_available) ? count : cb->samples_available;

    for (int i = 0; i < to_read; i++) {
        int pos = (cb->read_pos + i) % cb->size;
        data[i] = cb->data[pos];
    }

    pthread_mutex_unlock(&cb->mutex);
    return to_read;
}

int circular_buffer_get_latest(CircularBuffer *cb, float *data, int count) {
    pthread_mutex_lock(&cb->mutex);

    int available = cb->samples_available;
    if (available < count) {
        pthread_mutex_unlock(&cb->mutex);
        return 0;  // Not enough data yet
    }

    // Read from the end, going backwards
    int start_pos = (cb->write_pos - count + cb->size) % cb->size;

    for (int i = 0; i < count; i++) {
        int pos = (start_pos + i) % cb->size;
        data[i] = cb->data[pos];
    }

    pthread_mutex_unlock(&cb->mutex);
    return count;
}

void create_spectrogram(const char *wav_file, const char *output_file) {
    // Load audio
    sf_count_t num_frames;
    float *audio = load_audio(wav_file, &num_frames, SR);
    if (!audio) {
        return;
    }

    printf("Loaded audio: %ld frames at %d Hz\n", (long)num_frames, SR);

    // Calculate number of time frames
    int n_frames = 1 + (num_frames - N_FFT) / HOP_LENGTH;
    if (n_frames <= 0) {
        fprintf(stderr, "Audio file too short\n");
        free(audio);
        return;
    }

    // Allocate mel filterbank
    float **filterbank = (float**)malloc(N_MELS * sizeof(float*));
    for (int i = 0; i < N_MELS; i++) {
        filterbank[i] = (float*)calloc(N_FFT / 2 + 1, sizeof(float));
    }
    create_mel_filterbank(filterbank, N_MELS, N_FFT, SR);

    // Allocate spectrogram
    float **mel_spec = (float**)malloc(N_MELS * sizeof(float*));
    for (int i = 0; i < N_MELS; i++) {
        mel_spec[i] = (float*)calloc(n_frames, sizeof(float));
    }

    // FFTW setup
    float *fft_input = (float*)fftwf_malloc(N_FFT * sizeof(float));
    fftwf_complex *fft_output = (fftwf_complex*)fftwf_malloc((N_FFT / 2 + 1) * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N_FFT, fft_input, fft_output, FFTW_ESTIMATE);

    // Compute STFT and mel spectrogram
    for (int frame = 0; frame < n_frames; frame++) {
        int start = frame * HOP_LENGTH;

        // Copy frame with padding
        for (int i = 0; i < N_FFT; i++) {
            if (start + i < num_frames) {
                fft_input[i] = audio[start + i];
            } else {
                fft_input[i] = 0.0f;
            }
        }

        // Apply window
        apply_hanning_window(fft_input, N_FFT);

        // Execute FFT
        fftwf_execute(plan);

        // Compute power spectrum (matching librosa's power calculation)
        float *power_spec = (float*)malloc((N_FFT / 2 + 1) * sizeof(float));
        for (int i = 0; i < N_FFT / 2 + 1; i++) {
            float re = fft_output[i][0];
            float im = fft_output[i][1];
            // Scale by 1/n_fft for proper normalization
            power_spec[i] = (re * re + im * im) / (N_FFT * N_FFT);
        }

        // Apply mel filterbank
        for (int mel = 0; mel < N_MELS; mel++) {
            float mel_power = 0.0f;
            for (int i = 0; i < N_FFT / 2 + 1; i++) {
                mel_power += filterbank[mel][i] * power_spec[i];
            }
            mel_spec[mel][frame] = mel_power;
        }

        free(power_spec);
    }

    // Find max power for reference (librosa's ref=np.max)
    float max_power = 1e-10f;
    for (int mel = 0; mel < N_MELS; mel++) {
        for (int frame = 0; frame < n_frames; frame++) {
            if (mel_spec[mel][frame] > max_power) {
                max_power = mel_spec[mel][frame];
            }
        }
    }

    // Convert to dB (matching librosa.power_to_db)
    float amin = 1e-10f;
    float top_db = 80.0f;  // librosa default

    for (int mel = 0; mel < N_MELS; mel++) {
        for (int frame = 0; frame < n_frames; frame++) {
            // Add small value to avoid log(0)
            float power = fmaxf(mel_spec[mel][frame], amin);
            float ref = fmaxf(max_power, amin);
            mel_spec[mel][frame] = 10.0f * log10f(power / ref);
        }
    }

    // Apply top_db threshold
    float max_db = 0.0f;  // Since we use ref=np.max, max should be 0 dB
    float min_db = max_db - top_db;

    for (int mel = 0; mel < N_MELS; mel++) {
        for (int frame = 0; frame < n_frames; frame++) {
            if (mel_spec[mel][frame] < min_db) {
                mel_spec[mel][frame] = min_db;
            }
        }
    }

    // Find actual min/max for normalization
    float actual_min = 0.0f, actual_max = -1000.0f;
    for (int mel = 0; mel < N_MELS; mel++) {
        for (int frame = 0; frame < n_frames; frame++) {
            if (mel_spec[mel][frame] < actual_min) actual_min = mel_spec[mel][frame];
            if (mel_spec[mel][frame] > actual_max) actual_max = mel_spec[mel][frame];
        }
    }

    // Normalize to [0, 1]
    float range = actual_max - actual_min;
    if (range > 0.0f) {
        for (int mel = 0; mel < N_MELS; mel++) {
            for (int frame = 0; frame < n_frames; frame++) {
                mel_spec[mel][frame] = (mel_spec[mel][frame] - actual_min) / range;
            }
        }
    }

    // Create RGB image (flip vertically to match librosa)
    unsigned char *image = (unsigned char*)malloc(N_MELS * n_frames * 3);
    for (int y = 0; y < N_MELS; y++) {
        for (int x = 0; x < n_frames; x++) {
            // Flip vertically
            int flipped_y = N_MELS - 1 - y;
            float value = mel_spec[flipped_y][x];

            unsigned char rgb[3];
            apply_hot_colormap(value, rgb);

            int idx = (y * n_frames + x) * 3;
            image[idx + 0] = rgb[0];
            image[idx + 1] = rgb[1];
            image[idx + 2] = rgb[2];
        }
    }

    // Save PNG using fpng
    fpng::fpng_init();
    if (!fpng::fpng_encode_image_to_file(output_file, image, n_frames, N_MELS, 3)) {
        fprintf(stderr, "Error: Failed to save PNG file\n");
    } else {
        printf("Mel spectrogram saved to %s (%d x %d)\n", output_file, n_frames, N_MELS);
    }

    // Cleanup
    free(image);
    free(audio);
    fftwf_destroy_plan(plan);
    fftwf_free(fft_input);
    fftwf_free(fft_output);

    for (int i = 0; i < N_MELS; i++) {
        free(filterbank[i]);
        free(mel_spec[i]);
    }
    free(filterbank);
    free(mel_spec);
}

// libsoundio read callback
static void read_callback(struct SoundIoInStream *instream, int frame_count_min, int frame_count_max) {
    struct SoundIoChannelArea *areas;
    int err;

    int frames_left = frame_count_max;

    while (frames_left > 0) {
        int frame_count = frames_left;

        if ((err = soundio_instream_begin_read(instream, &areas, &frame_count))) {
            fprintf(stderr, "Error reading from stream: %s\n", soundio_strerror(err));
            return;
        }

        if (!frame_count)
            break;

        if (!areas) {
            // Silence
            frames_left -= frame_count;
        } else {
            // Use pre-allocated buffer (NO malloc in real-time callback!)
            // Ensure buffer is large enough (should be allocated for max possible frame_count)
            if (frame_count > g_stream_state.audio_conversion_buffer_size) {
                fprintf(stderr, "Warning: frame_count %d exceeds buffer size %d\n",
                        frame_count, g_stream_state.audio_conversion_buffer_size);
                frame_count = g_stream_state.audio_conversion_buffer_size;
            }

            for (int frame = 0; frame < frame_count; frame++) {
                float sample = 0.0f;

                // Average all channels to mono
                for (int ch = 0; ch < instream->layout.channel_count; ch++) {
                    float *ptr = (float*)(areas[ch].ptr + areas[ch].step * frame);
                    sample += *ptr;
                }
                sample /= instream->layout.channel_count;

                g_stream_state.audio_conversion_buffer[frame] = sample;
            }

            circular_buffer_write(&g_stream_state.buffer, g_stream_state.audio_conversion_buffer, frame_count);

            frames_left -= frame_count;
        }

        if ((err = soundio_instream_end_read(instream))) {
            fprintf(stderr, "Error ending read: %s\n", soundio_strerror(err));
            return;
        }
    }
}

static void overflow_callback(struct SoundIoInStream *instream) {
    static int count = 0;
    fprintf(stderr, "Overflow %d\n", ++count);
}

// GLFW error callback
static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Framebuffer size callback
static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Vertex shader with aspect ratio correction
const char* vertex_shader_source = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "uniform vec2 scale;\n"
    "void main() {\n"
    "    gl_Position = vec4(aPos.x * scale.x, aPos.y * scale.y, 0.0, 1.0);\n"
    "    TexCoord = aTexCoord;\n"
    "}\0";

// Simple fragment shader to display texture
const char* fragment_shader_source = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main() {\n"
    "    FragColor = texture(texture1, TexCoord);\n"
    "}\0";

// Initialize OpenGL context and FBO
int init_gl_context(GLContext* ctx, int width, int height) {
    pthread_mutex_init(&ctx->mutex, NULL);
    ctx->pending_image = NULL;
    ctx->upload_buffer = NULL;
    ctx->upload_buffer_size = 0;
    ctx->has_pending_update = 0;

    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    ctx->window = glfwCreateWindow(1024, 768, "Real-time Mel Spectrogram", NULL, NULL);
    if (!ctx->window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(ctx->window);
    glfwSetFramebufferSizeCallback(ctx->window, framebuffer_size_callback);

    // Enable vsync for smooth frame pacing (adapts to monitor refresh rate)
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    ctx->tex_width = width;
    ctx->tex_height = height;

    // Create FBO texture - initialize to black to prevent white flashing
    glGenTextures(1, &ctx->texture);
    glBindTexture(GL_TEXTURE_2D, ctx->texture);

    // Create black initialization data
    unsigned char* black_data = (unsigned char*)calloc(width * height * 3, sizeof(unsigned char));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, black_data);
    free(black_data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create FBO
    glGenFramebuffers(1, &ctx->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ctx->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctx->texture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO not complete\n");
        return -1;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Compile shaders
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);

    ctx->shader_program = glCreateProgram();
    glAttachShader(ctx->shader_program, vertex_shader);
    glAttachShader(ctx->shader_program, fragment_shader);
    glLinkProgram(ctx->shader_program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // Setup fullscreen quad with flipped V coordinates (fix upside-down)
    float vertices[] = {
        // positions   // texCoords (V flipped: 0 at top, 1 at bottom)
        -1.0f,  1.0f,  0.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f
    };

    glGenVertexArrays(1, &ctx->vao);
    glGenBuffers(1, &ctx->vbo);
    glBindVertexArray(ctx->vao);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    printf("OpenGL context initialized: %dx%d FBO texture\n", width, height);
    return 0;
}

// Queue image data for FBO texture update (called from processing thread)
void queue_fbo_texture_update(GLContext* ctx, unsigned char* image_data, int width, int height) {
    if (!ctx || !image_data || width <= 0 || height <= 0) {
        fprintf(stderr, "Invalid parameters to queue_fbo_texture_update\n");
        return;
    }

    int size = width * height * 3;
    unsigned char* new_image = (unsigned char*)malloc(size);
    if (!new_image) {
        fprintf(stderr, "Failed to allocate pending image buffer\n");
        return;
    }

    // Copy data before locking to minimize time in critical section
    memcpy(new_image, image_data, size);

    pthread_mutex_lock(&ctx->mutex);

    // Free previous pending image if any
    if (ctx->pending_image) {
        free(ctx->pending_image);
        ctx->pending_image = NULL;
    }

    // Set new pending image
    ctx->pending_image = new_image;
    ctx->pending_width = width;
    ctx->pending_height = height;
    ctx->has_pending_update = 1;

    pthread_mutex_unlock(&ctx->mutex);
}

// Apply pending texture update (called from main thread with active GL context)
void apply_pending_texture_update(GLContext* ctx) {
    if (!ctx) return;

    pthread_mutex_lock(&ctx->mutex);

    // Check if there's an update pending
    if (!ctx->has_pending_update || !ctx->pending_image) {
        pthread_mutex_unlock(&ctx->mutex);
        return;
    }

    int upload_width = ctx->pending_width;
    int upload_height = ctx->pending_height;
    int size = upload_width * upload_height * 3;

    // Validate dimensions
    if (upload_width <= 0 || upload_height <= 0 ||
        upload_width > ctx->tex_width || upload_height > ctx->tex_height) {
        fprintf(stderr, "Invalid texture dimensions: %dx%d\n", upload_width, upload_height);
        pthread_mutex_unlock(&ctx->mutex);
        return;
    }

    // Allocate or reallocate upload buffer if needed
    if (ctx->upload_buffer == NULL || ctx->upload_buffer_size < size) {
        unsigned char* new_buffer = (unsigned char*)realloc(ctx->upload_buffer, size);
        if (!new_buffer) {
            fprintf(stderr, "Failed to allocate upload buffer\n");
            pthread_mutex_unlock(&ctx->mutex);
            return;
        }
        ctx->upload_buffer = new_buffer;
        ctx->upload_buffer_size = size;
    }

    // Copy data from pending to upload buffer
    memcpy(ctx->upload_buffer, ctx->pending_image, size);
    ctx->has_pending_update = 0;

    pthread_mutex_unlock(&ctx->mutex);

    // Upload to GL texture (outside mutex to avoid blocking processing thread)
    glBindTexture(GL_TEXTURE_2D, ctx->texture);

    // Set proper pixel alignment
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, upload_width, upload_height,
                    GL_RGB, GL_UNSIGNED_BYTE, ctx->upload_buffer);

    // Check for GL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "GL error in texture upload: 0x%x\n", err);
    }
}

// Render FBO texture to screen
void render_fbo_to_screen(GLContext* ctx) {
    // Apply any pending texture updates first
    apply_pending_texture_update(ctx);

    int window_width, window_height;
    glfwGetFramebufferSize(ctx->window, &window_width, &window_height);

    glViewport(0, 0, window_width, window_height);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(ctx->shader_program);

    // Calculate aspect ratio preserving scale
    float tex_aspect = (float)ctx->tex_width / (float)ctx->tex_height;
    float win_aspect = (float)window_width / (float)window_height;
    float scale_x = 1.0f, scale_y = 1.0f;

    if (win_aspect > tex_aspect) {
        // Window is wider than texture - scale width down
        scale_x = tex_aspect / win_aspect;
    } else {
        // Window is taller than texture - scale height down
        scale_y = win_aspect / tex_aspect;
    }

    // Set scale uniform
    GLint scale_loc = glGetUniformLocation(ctx->shader_program, "scale");
    glUniform2f(scale_loc, scale_x, scale_y);

    glBindTexture(GL_TEXTURE_2D, ctx->texture);
    glBindVertexArray(ctx->vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(ctx->window);
}

// Create output directory if it doesn't exist
void ensure_output_directory() {
    struct stat st = {0};
    if (stat(OUTPUT_DIR, &st) == -1) {
        mkdir(OUTPUT_DIR, 0755);
        printf("Created output directory: %s\n", OUTPUT_DIR);
    }
}

// Generate spectrogram from audio buffer (streaming version with 5s rolling window)
void *streaming_spectrogram_thread(void *arg) {
    StreamState *state = (StreamState*)arg;

    // Ensure output directory exists
    ensure_output_directory();

    // Calculate number of frames needed for rolling window (5 seconds)
    // Each frame represents HOP_LENGTH samples of unique data
    int rolling_frames = (int)((ROLLING_WINDOW_DURATION * SR) / HOP_LENGTH);
    printf("Rolling window: %.1f seconds (%d frames)\n", ROLLING_WINDOW_DURATION, rolling_frames);

    // Initialize rolling buffer for mel spectrogram frames
    mel_rolling_buffer_init(&state->mel_buffer, rolling_frames, N_MELS);

    // Allocate mel filterbank (reused for all frames)
    float **filterbank = (float**)malloc(N_MELS * sizeof(float*));
    for (int i = 0; i < N_MELS; i++) {
        filterbank[i] = (float*)calloc(N_FFT / 2 + 1, sizeof(float));
    }
    create_mel_filterbank(filterbank, N_MELS, N_FFT, SR);

    // FFTW setup (FFTW_MEASURE optimizes FFT plan for better performance)
    float *fft_input = (float*)fftwf_malloc(N_FFT * sizeof(float));
    fftwf_complex *fft_output = (fftwf_complex*)fftwf_malloc((N_FFT / 2 + 1) * sizeof(fftwf_complex));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N_FFT, fft_input, fft_output, FFTW_MEASURE);

    // Buffer for single frame of audio (N_FFT samples for FFT window)
    float *audio_frame = (float*)malloc(N_FFT * sizeof(float));

    // Pre-allocate power spectrum buffer and mel frame buffer (reused)
    float *power_spec = (float*)malloc((N_FFT / 2 + 1) * sizeof(float));
    float *mel_frame = (float*)malloc(N_MELS * sizeof(float));

    // Pre-allocate db_values buffer to avoid per-frame malloc/free (CRITICAL for reducing jitter)
    float **db_values = (float**)malloc(rolling_frames * sizeof(float*));
    for (int i = 0; i < rolling_frames; i++) {
        db_values[i] = (float*)malloc(N_MELS * sizeof(float));
    }

    // Pre-allocate image buffer to avoid per-frame malloc/free
    int max_image_size = N_MELS * rolling_frames * 3;
    unsigned char *image = (unsigned char*)malloc(max_image_size);

    fpng::fpng_init();

    // Initialize audio read position
    state->audio_read_pos = 0;

    // Precise timing for frame pacing
    struct timespec frame_start, frame_end;
    const long target_frame_time_us = (long)(HOP_LENGTH * 1000000.0 / SR);  // Target frame time in microseconds

    while (state->running) {
        clock_gettime(CLOCK_MONOTONIC, &frame_start);

        // Check if we have enough audio data for a new frame
        pthread_mutex_lock(&state->buffer.mutex);
        int available = state->buffer.samples_available;
        pthread_mutex_unlock(&state->buffer.mutex);

        // We need at least N_FFT samples to process a frame
        if (available < N_FFT) {
            usleep(1000);  // Sleep 1ms and retry (shorter sleep for better responsiveness)
            continue;
        }

        // Read N_FFT samples from the circular buffer at current read position
        // We advance by HOP_LENGTH for each frame, so overlap is handled naturally
        pthread_mutex_lock(&state->buffer.mutex);
        int buffer_size = state->buffer.size;
        int write_pos = state->buffer.write_pos;

        // Calculate starting position (N_FFT samples back from write position)
        int start_pos = (write_pos - N_FFT + buffer_size) % buffer_size;

        for (int i = 0; i < N_FFT; i++) {
            int pos = (start_pos + i) % buffer_size;
            audio_frame[i] = state->buffer.data[pos];
        }
        pthread_mutex_unlock(&state->buffer.mutex);

        // Apply Hanning window
        apply_hanning_window(audio_frame, N_FFT);

        // Execute FFT
        memcpy(fft_input, audio_frame, N_FFT * sizeof(float));
        fftwf_execute(plan);

        // Compute power spectrum
        for (int i = 0; i < N_FFT / 2 + 1; i++) {
            float re = fft_output[i][0];
            float im = fft_output[i][1];
            power_spec[i] = (re * re + im * im) / (N_FFT * N_FFT);
        }

        // Apply mel filterbank to get one frame of mel spectrogram
        // Store as power values (not dB yet - we'll convert later with proper reference)
        for (int mel = 0; mel < N_MELS; mel++) {
            float mel_power = 0.0f;
            for (int i = 0; i < N_FFT / 2 + 1; i++) {
                mel_power += filterbank[mel][i] * power_spec[i];
            }
            mel_frame[mel] = mel_power;
        }

        // Add this frame to the rolling buffer
        mel_rolling_buffer_add_frame(&state->mel_buffer, mel_frame, N_MELS);

        // Now generate output image from the entire rolling buffer
        int output_width = state->mel_buffer.current_frames;

        if (output_width > 0) {
            // Step 1: Find max power across all frames (for dB reference)
            float max_power = 1e-10f;
            for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
                int buffer_idx = (state->mel_buffer.write_pos - output_width + frame_idx + state->mel_buffer.max_frames) % state->mel_buffer.max_frames;
                for (int mel = 0; mel < N_MELS; mel++) {
                    float power = state->mel_buffer.frames[buffer_idx][mel];
                    if (power > max_power) max_power = power;
                }
            }

            // Step 2: Convert to dB and find min/max for normalization
            float amin = 1e-10f;
            float top_db = 80.0f;

            // Use pre-allocated db_values buffer (no malloc needed)
            float min_db = 0.0f, max_db = 0.0f;
            for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
                int buffer_idx = (state->mel_buffer.write_pos - output_width + frame_idx + state->mel_buffer.max_frames) % state->mel_buffer.max_frames;
                for (int mel = 0; mel < N_MELS; mel++) {
                    float power = fmaxf(state->mel_buffer.frames[buffer_idx][mel], amin);
                    float ref = fmaxf(max_power, amin);
                    float db = 10.0f * log10f(power / ref);

                    // Apply top_db threshold
                    if (db < -top_db) db = -top_db;

                    db_values[frame_idx][mel] = db;
                    if (db < min_db) min_db = db;
                    if (db > max_db) max_db = db;
                }
            }

            // Step 3: Normalize to [0, 1] and create RGB image
            float range = max_db - min_db;
            if (range < 0.001f) range = 0.001f;

            // Use pre-allocated image buffer (no malloc needed)
            for (int y = 0; y < N_MELS; y++) {
                for (int x = 0; x < output_width; x++) {
                    // Flip vertically for display
                    int flipped_y = N_MELS - 1 - y;
                    float db_value = db_values[x][flipped_y];

                    // Normalize to [0, 1]
                    float normalized = (db_value - min_db) / range;
                    normalized = fminf(1.0f, fmaxf(0.0f, normalized));

                    unsigned char rgb[3];
                    // Use blue colormap for system audio, hot colormap for microphone
                    if (state->use_system_audio) {
                        apply_blue_colormap(normalized, rgb);
                    } else {
                        apply_hot_colormap(normalized, rgb);
                    }

                    int idx = (y * output_width + x) * 3;
                    image[idx + 0] = rgb[0];
                    image[idx + 1] = rgb[1];
                    image[idx + 2] = rgb[2];
                }
            }

            // Debug: Check image data and color distribution
            if (state->frame_counter % 60 == 0) {  // Every 60 frames
                printf("\nDebug: range=%.2f, min_db=%.2f, max_db=%.2f, max_power=%.2e\n",
                       range, min_db, max_db, max_power);

                // Check several pixels to see color distribution
                int mid_pixel = (N_MELS * output_width / 2) * 3;
                printf("First pixel RGB: [%d, %d, %d]\n", image[0], image[1], image[2]);
                printf("Mid pixel RGB: [%d, %d, %d]\n", image[mid_pixel], image[mid_pixel+1], image[mid_pixel+2]);
                printf("Last pixel RGB: [%d, %d, %d]\n",
                       image[(N_MELS * output_width - 1) * 3],
                       image[(N_MELS * output_width - 1) * 3 + 1],
                       image[(N_MELS * output_width - 1) * 3 + 2]);

                // Test colormap directly
                unsigned char test_rgb[3];
                if (state->use_system_audio) {
                    apply_blue_colormap(0.0f, test_rgb);
                    printf("Blue colormap test: 0.0 -> [%d,%d,%d], ", test_rgb[0], test_rgb[1], test_rgb[2]);
                    apply_blue_colormap(0.5f, test_rgb);
                    printf("0.5 -> [%d,%d,%d], ", test_rgb[0], test_rgb[1], test_rgb[2]);
                    apply_blue_colormap(1.0f, test_rgb);
                    printf("1.0 -> [%d,%d,%d]\n", test_rgb[0], test_rgb[1], test_rgb[2]);
                } else {
                    apply_hot_colormap(0.0f, test_rgb);
                    printf("Hot colormap test: 0.0 -> [%d,%d,%d], ", test_rgb[0], test_rgb[1], test_rgb[2]);
                    apply_hot_colormap(0.5f, test_rgb);
                    printf("0.5 -> [%d,%d,%d], ", test_rgb[0], test_rgb[1], test_rgb[2]);
                    apply_hot_colormap(1.0f, test_rgb);
                    printf("1.0 -> [%d,%d,%d]\n", test_rgb[0], test_rgb[1], test_rgb[2]);
                }
            }

            // Queue image data for FBO texture update (will be applied on main thread)
            queue_fbo_texture_update(&state->gl_context, image, output_width, N_MELS);

            // Save PNG every 60 frames for color verification
            // DISABLED: PNG saving turned off
            /*
            if (state->frame_counter % 60 == 0) {
                char filename[512];
                snprintf(filename, sizeof(filename), "%s/%d_rolling_spectrogram.png", OUTPUT_DIR, state->frame_counter);
                fpng::fpng_encode_image_to_file(filename, image, output_width, N_MELS, 3);
                printf("Saved PNG: %s\n", filename);
            }
            */

            printf("\rFrame %d - Rolling window: %.2fs (%d frames, %d x %d)",
                   state->frame_counter,
                   (float)output_width * HOP_LENGTH / SR,
                   output_width,
                   output_width,
                   N_MELS);
            fflush(stdout);
        }

        state->frame_counter++;

        // Precise timing compensation: sleep only for remaining time in frame period
        clock_gettime(CLOCK_MONOTONIC, &frame_end);
        long elapsed_us = (frame_end.tv_sec - frame_start.tv_sec) * 1000000L +
                          (frame_end.tv_nsec - frame_start.tv_nsec) / 1000L;

        long sleep_us = target_frame_time_us - elapsed_us;
        if (sleep_us > 0) {
            usleep(sleep_us);
        }
        // If we took longer than target, don't sleep (process next frame immediately)
    }

    // Cleanup
    mel_rolling_buffer_free(&state->mel_buffer);
    free(audio_frame);
    free(power_spec);
    free(mel_frame);

    // Free pre-allocated db_values buffer
    for (int i = 0; i < rolling_frames; i++) {
        free(db_values[i]);
    }
    free(db_values);

    // Free pre-allocated image buffer
    free(image);

    fftwf_destroy_plan(plan);
    fftwf_free(fft_input);
    fftwf_free(fft_output);

    for (int i = 0; i < N_MELS; i++) {
        free(filterbank[i]);
    }
    free(filterbank);

    printf("\nStreaming stopped.\n");
    return NULL;
}

// Find a loopback/monitor device for system audio capture
int find_loopback_device(struct SoundIo *soundio) {
    int input_count = soundio_input_device_count(soundio);

    // Keywords that indicate a loopback/monitor device
    const char *loopback_keywords[] = {
        "monitor",      // PulseAudio monitor sources
        "loopback",     // Generic loopback devices
        "Stereo Mix",   // Windows stereo mix
        "Wave Out",     // Windows wave out mix
        "What U Hear",  // Creative sound cards
        "Rec. Playback" // Some Linux ALSA devices
    };
    int num_keywords = sizeof(loopback_keywords) / sizeof(loopback_keywords[0]);

    printf("Searching for system audio loopback device...\n");
    printf("Available input devices:\n");

    for (int i = 0; i < input_count; i++) {
        struct SoundIoDevice *device = soundio_get_input_device(soundio, i);
        if (!device) continue;

        printf("  [%d] %s\n", i, device->name);

        // Check if device name contains any loopback keywords (case-insensitive)
        for (int k = 0; k < num_keywords; k++) {
            if (strcasestr(device->name, loopback_keywords[k]) != NULL) {
                printf("Found loopback device: %s\n", device->name);
                int index = i;
                soundio_device_unref(device);
                return index;
            }
        }

        soundio_device_unref(device);
    }

    return -1;  // No loopback device found
}

// Start microphone streaming
int start_microphone_stream(const char *output_file, int use_system_audio) {
    struct SoundIo *soundio = soundio_create();
    if (!soundio) {
        fprintf(stderr, "Error: Out of memory\n");
        return 1;
    }

    int err = soundio_connect(soundio);
    if (err) {
        fprintf(stderr, "Error connecting: %s\n", soundio_strerror(err));
        soundio_destroy(soundio);
        return 1;
    }

    soundio_flush_events(soundio);

    int device_index;

    if (use_system_audio) {
        // Find a loopback/monitor device for system audio
        device_index = find_loopback_device(soundio);
        if (device_index < 0) {
            fprintf(stderr, "Error: No system audio loopback device found\n");
            fprintf(stderr, "\nOn Linux with PulseAudio, you may need to:\n");
            fprintf(stderr, "  1. Check available monitor sources: pactl list sources | grep -i monitor\n");
            fprintf(stderr, "  2. Load a loopback module: pactl load-module module-loopback\n");
            fprintf(stderr, "\nOn Windows, enable 'Stereo Mix' in sound settings.\n");
            soundio_destroy(soundio);
            return 1;
        }
    } else {
        // Use default microphone input
        device_index = soundio_default_input_device_index(soundio);
        if (device_index < 0) {
            fprintf(stderr, "Error: No input device found\n");
            soundio_destroy(soundio);
            return 1;
        }
    }

    struct SoundIoDevice *device = soundio_get_input_device(soundio, device_index);
    if (!device) {
        fprintf(stderr, "Error: Out of memory\n");
        soundio_destroy(soundio);
        return 1;
    }

    printf("Using input device: %s\n", device->name);

    struct SoundIoInStream *instream = soundio_instream_create(device);
    if (!instream) {
        fprintf(stderr, "Error: Out of memory\n");
        soundio_device_unref(device);
        soundio_destroy(soundio);
        return 1;
    }

    instream->format = SoundIoFormatFloat32NE;
    instream->sample_rate = SR;
    instream->read_callback = read_callback;
    instream->overflow_callback = overflow_callback;
    instream->software_latency = 0.01;  // 10ms latency for real-time feel

    if ((err = soundio_instream_open(instream))) {
        fprintf(stderr, "Error opening stream: %s\n", soundio_strerror(err));
        soundio_instream_destroy(instream);
        soundio_device_unref(device);
        soundio_destroy(soundio);
        return 1;
    }

    // Initialize circular buffer
    int buffer_size = (int)(SR * BUFFER_DURATION);
    circular_buffer_init(&g_stream_state.buffer, buffer_size);
    g_stream_state.sample_rate = SR;
    g_stream_state.running = 1;
    g_stream_state.output_file = output_file;
    g_stream_state.frame_counter = 0;
    g_stream_state.use_system_audio = use_system_audio;

    // Pre-allocate audio conversion buffer for read callback (avoid malloc in RT context)
    // Use 2x the period size for safety (typical period ~512 frames at 22050 Hz)
    g_stream_state.audio_conversion_buffer_size = 2048;
    g_stream_state.audio_conversion_buffer = (float*)malloc(g_stream_state.audio_conversion_buffer_size * sizeof(float));
    if (!g_stream_state.audio_conversion_buffer) {
        fprintf(stderr, "Failed to allocate audio conversion buffer\n");
        circular_buffer_free(&g_stream_state.buffer);
        return 1;
    }

    // Initialize OpenGL context and FBO
    int rolling_frames = (int)((ROLLING_WINDOW_DURATION * SR) / HOP_LENGTH);
    if (init_gl_context(&g_stream_state.gl_context, rolling_frames, N_MELS) != 0) {
        fprintf(stderr, "Failed to initialize OpenGL context\n");
        return 1;
    }

    // Start processing thread
    pthread_create(&g_stream_state.processing_thread, NULL, streaming_spectrogram_thread, &g_stream_state);

    if ((err = soundio_instream_start(instream))) {
        fprintf(stderr, "Error starting stream: %s\n", soundio_strerror(err));
        soundio_instream_destroy(instream);
        soundio_device_unref(device);
        soundio_destroy(soundio);
        return 1;
    }

    printf("Recording from microphone... Close window or press ESC to stop\n");

    // Main render loop - display FBO texture
    while (!glfwWindowShouldClose(g_stream_state.gl_context.window)) {
        // Handle GLFW events
        glfwPollEvents();

        // Check for ESC key
        if (glfwGetKey(g_stream_state.gl_context.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(g_stream_state.gl_context.window, 1);
        }

        // Render FBO texture to screen (vsync will pace frames at display refresh rate)
        render_fbo_to_screen(&g_stream_state.gl_context);

        // Process audio events
        soundio_flush_events(soundio);
    }

    // Cleanup
    g_stream_state.running = 0;
    pthread_join(g_stream_state.processing_thread, NULL);
    circular_buffer_free(&g_stream_state.buffer);
    free(g_stream_state.audio_conversion_buffer);
    soundio_instream_destroy(instream);
    soundio_device_unref(device);
    soundio_destroy(soundio);
    glfwTerminate();

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s [--mic|--system-audio] <input.wav> [output.png]\n", argv[0]);
        printf("  --mic              Use microphone input (real-time streaming)\n");
        printf("  --system-audio     Use system audio/loopback (real-time streaming)\n");
        printf("  <input.wav>        Input WAV file (file mode)\n");
        printf("  [output.png]       Output PNG file (default: spectrogram.png)\n");
        return 1;
    }

    // Check for streaming flags
    int use_microphone = 0;
    int use_system_audio = 0;
    int arg_offset = 1;

    if (strcmp(argv[1], "--mic") == 0) {
        use_microphone = 1;
        arg_offset = 2;
    } else if (strcmp(argv[1], "--system-audio") == 0) {
        use_system_audio = 1;
        arg_offset = 2;
    }

    if (use_microphone || use_system_audio) {
        const char *output_png = (argc > arg_offset) ? argv[arg_offset] : "spectrogram_stream.png";
        return start_microphone_stream(output_png, use_system_audio);
    } else {
        const char *input_wav = argv[arg_offset];
        const char *output_png = (argc > arg_offset + 1) ? argv[arg_offset + 1] : "spectrogram.png";
        create_spectrogram(input_wav, output_png);
        return 0;
    }
}
