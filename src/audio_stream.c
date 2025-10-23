#include "audio_stream.h"
#include "melspec.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <soundio/soundio.h>
#include <strings.h>

#define BUFFER_DURATION 2.0f
#define ROLLING_WINDOW_DURATION 5.0f

// Circular buffer for streaming audio
typedef struct {
    float *data;
    int size;
    int write_pos;
    int read_pos;
    pthread_mutex_t mutex;
    int samples_available;
} CircularBuffer;

// Audio stream context (internal)
struct AudioStreamContext {
    CircularBuffer buffer;
    int sample_rate;
    int running;
    pthread_t processing_thread;
    MelRollingBuffer mel_buffer;
    int use_system_audio;
    float *audio_conversion_buffer;
    int audio_conversion_buffer_size;
    AudioFrameCallback callback;
    void* user_data;

    // libsoundio objects
    struct SoundIo *soundio;
    struct SoundIoDevice *device;
    struct SoundIoInStream *instream;

    int frame_counter;
};

// Hot colormap - matching matplotlib's implementation exactly
static void apply_hot_colormap(float value, unsigned char *rgb) {
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
static void apply_blue_colormap(float value, unsigned char *rgb) {
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

// Circular buffer functions
static void circular_buffer_init(CircularBuffer *cb, int size) {
    cb->data = (float*)calloc(size, sizeof(float));
    cb->size = size;
    cb->write_pos = 0;
    cb->read_pos = 0;
    cb->samples_available = 0;
    pthread_mutex_init(&cb->mutex, NULL);
}

static void circular_buffer_free(CircularBuffer *cb) {
    free(cb->data);
    pthread_mutex_destroy(&cb->mutex);
}

static void circular_buffer_write(CircularBuffer *cb, const float *data, int count) {
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

// libsoundio read callback
static void read_callback(struct SoundIoInStream *instream, int frame_count_min, int frame_count_max) {
    struct SoundIoChannelArea *areas;
    int err;
    AudioStreamContext *ctx = (AudioStreamContext*)instream->userdata;

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
            if (frame_count > ctx->audio_conversion_buffer_size) {
                fprintf(stderr, "Warning: frame_count %d exceeds buffer size %d\n",
                        frame_count, ctx->audio_conversion_buffer_size);
                frame_count = ctx->audio_conversion_buffer_size;
            }

            for (int frame = 0; frame < frame_count; frame++) {
                float sample = 0.0f;

                // Average all channels to mono
                for (int ch = 0; ch < instream->layout.channel_count; ch++) {
                    float *ptr = (float*)(areas[ch].ptr + areas[ch].step * frame);
                    sample += *ptr;
                }
                sample /= instream->layout.channel_count;

                ctx->audio_conversion_buffer[frame] = sample;
            }

            circular_buffer_write(&ctx->buffer, ctx->audio_conversion_buffer, frame_count);

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

// Streaming spectrogram processing thread
// RUNS IN WORKER THREAD - All mel spectrogram generation happens here
// Main thread only handles FBO/GL operations via callback
static void *streaming_spectrogram_thread(void *arg) {
    AudioStreamContext *ctx = (AudioStreamContext*)arg;

    // Calculate number of frames needed for rolling window
    int rolling_frames = (int)((ROLLING_WINDOW_DURATION * MELSPEC_SR) / MELSPEC_HOP_LENGTH);
    printf("Rolling window: %.1f seconds (%d frames)\n", ROLLING_WINDOW_DURATION, rolling_frames);

    // Initialize rolling buffer for mel spectrogram frames
    melspec_rolling_buffer_init(&ctx->mel_buffer, rolling_frames, MELSPEC_N_MELS);

    // Initialize mel spectrogram context
    MelSpectrogramContext* melspec_ctx = melspec_init(MELSPEC_N_FFT, MELSPEC_N_MELS, MELSPEC_SR);
    if (!melspec_ctx) {
        fprintf(stderr, "Failed to initialize mel spectrogram context\n");
        return NULL;
    }

    // Buffer for single frame of audio
    float *audio_frame = (float*)malloc(MELSPEC_N_FFT * sizeof(float));

    // Pre-allocate mel frame buffer (reused)
    float *mel_frame = (float*)malloc(MELSPEC_N_MELS * sizeof(float));

    // Pre-allocate db_values buffer
    float **db_values = (float**)malloc(rolling_frames * sizeof(float*));
    for (int i = 0; i < rolling_frames; i++) {
        db_values[i] = (float*)malloc(MELSPEC_N_MELS * sizeof(float));
    }

    // Pre-allocate image buffer
    int max_image_size = MELSPEC_N_MELS * rolling_frames * 3;
    unsigned char *image = (unsigned char*)malloc(max_image_size);

    // Precise timing for frame pacing
    struct timespec frame_start, frame_end;
    const long target_frame_time_us = (long)(MELSPEC_HOP_LENGTH * 1000000.0 / MELSPEC_SR);

    while (ctx->running) {
        clock_gettime(CLOCK_MONOTONIC, &frame_start);

        // Check if we have enough audio data for a new frame
        pthread_mutex_lock(&ctx->buffer.mutex);
        int available = ctx->buffer.samples_available;
        pthread_mutex_unlock(&ctx->buffer.mutex);

        // We need at least N_FFT samples to process a frame
        if (available < MELSPEC_N_FFT) {
            usleep(1000);  // Sleep 1ms and retry
            continue;
        }

        // Read N_FFT samples from the circular buffer
        pthread_mutex_lock(&ctx->buffer.mutex);
        int buffer_size = ctx->buffer.size;
        int write_pos = ctx->buffer.write_pos;

        // Calculate starting position (N_FFT samples back from write position)
        int start_pos = (write_pos - MELSPEC_N_FFT + buffer_size) % buffer_size;

        for (int i = 0; i < MELSPEC_N_FFT; i++) {
            int pos = (start_pos + i) % buffer_size;
            audio_frame[i] = ctx->buffer.data[pos];
        }
        pthread_mutex_unlock(&ctx->buffer.mutex);

        // Apply Hanning window
        melspec_apply_hanning_window(audio_frame, MELSPEC_N_FFT);

        // Process frame to get mel coefficients
        melspec_process_frame(melspec_ctx, audio_frame, mel_frame);

        // Add this frame to the rolling buffer
        melspec_rolling_buffer_add_frame(&ctx->mel_buffer, mel_frame, MELSPEC_N_MELS);

        // Generate output image from the entire rolling buffer
        int output_width = ctx->mel_buffer.current_frames;

        if (output_width > 0) {
            // Step 1: Find max power across all frames
            float max_power = 1e-10f;
            for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
                int buffer_idx = (ctx->mel_buffer.write_pos - output_width + frame_idx + ctx->mel_buffer.max_frames) % ctx->mel_buffer.max_frames;
                for (int mel = 0; mel < MELSPEC_N_MELS; mel++) {
                    float power = ctx->mel_buffer.frames[buffer_idx][mel];
                    if (power > max_power) max_power = power;
                }
            }

            // Step 2: Convert to dB and find min/max for normalization
            float amin = 1e-10f;
            float top_db = 80.0f;

            float min_db = 0.0f, max_db = 0.0f;
            for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
                int buffer_idx = (ctx->mel_buffer.write_pos - output_width + frame_idx + ctx->mel_buffer.max_frames) % ctx->mel_buffer.max_frames;
                for (int mel = 0; mel < MELSPEC_N_MELS; mel++) {
                    float power = fmaxf(ctx->mel_buffer.frames[buffer_idx][mel], amin);
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

            for (int y = 0; y < MELSPEC_N_MELS; y++) {
                for (int x = 0; x < output_width; x++) {
                    // Flip vertically for display
                    int flipped_y = MELSPEC_N_MELS - 1 - y;
                    float db_value = db_values[x][flipped_y];

                    // Normalize to [0, 1]
                    float normalized = (db_value - min_db) / range;
                    normalized = fminf(1.0f, fmaxf(0.0f, normalized));

                    unsigned char rgb[3];
                    // Use blue colormap for system audio, hot colormap for microphone
                    if (ctx->use_system_audio) {
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

            // Call callback with image data
            if (ctx->callback) {
                ctx->callback(image, output_width, MELSPEC_N_MELS, ctx->user_data);
            }

            printf("\rFrame %d - Rolling window: %.2fs (%d frames, %d x %d)",
                   ctx->frame_counter,
                   (float)output_width * MELSPEC_HOP_LENGTH / MELSPEC_SR,
                   output_width,
                   output_width,
                   MELSPEC_N_MELS);
            fflush(stdout);
        }

        ctx->frame_counter++;

        // Precise timing compensation
        clock_gettime(CLOCK_MONOTONIC, &frame_end);
        long elapsed_us = (frame_end.tv_sec - frame_start.tv_sec) * 1000000L +
                          (frame_end.tv_nsec - frame_start.tv_nsec) / 1000L;

        long sleep_us = target_frame_time_us - elapsed_us;
        if (sleep_us > 0) {
            usleep(sleep_us);
        }
    }

    // Cleanup
    melspec_rolling_buffer_free(&ctx->mel_buffer);
    free(audio_frame);
    free(mel_frame);

    for (int i = 0; i < rolling_frames; i++) {
        free(db_values[i]);
    }
    free(db_values);
    free(image);

    melspec_free(melspec_ctx);

    printf("\nStreaming stopped.\n");
    return NULL;
}

// Find a loopback/monitor device for system audio capture
// Prioritizes the monitor source for the default output device
static int find_loopback_device(struct SoundIo *soundio) {
    int input_count = soundio_input_device_count(soundio);

    printf("Searching for system audio loopback device...\n");

    // First, try to find the default output device to get its monitor
    int default_output_index = soundio_default_output_device_index(soundio);
    if (default_output_index >= 0) {
        struct SoundIoDevice *output_device = soundio_get_output_device(soundio, default_output_index);
        if (output_device) {
            printf("Default output device: %s\n", output_device->name);
            printf("Searching for corresponding monitor source...\n");

            // Look for a monitor device that matches the output device
            // Common patterns:
            // - PulseAudio/PipeWire: "device_name.monitor"
            // - Some systems: "Monitor of device_name"
            for (int i = 0; i < input_count; i++) {
                struct SoundIoDevice *input_device = soundio_get_input_device(soundio, i);
                if (!input_device) continue;

                // Check if this monitor corresponds to the default output
                if (strcasestr(input_device->name, output_device->name) != NULL &&
                    strcasestr(input_device->name, "monitor") != NULL) {
                    printf("Found matching monitor for default output: %s\n", input_device->name);
                    int index = i;
                    soundio_device_unref(input_device);
                    soundio_device_unref(output_device);
                    return index;
                }

                soundio_device_unref(input_device);
            }

            soundio_device_unref(output_device);
            printf("No monitor found for default output, searching for any monitor device...\n");
        }
    }

    // Fallback: search for any loopback/monitor device
    const char *loopback_keywords[] = {
        "monitor",
        "loopback",
        "Stereo Mix",
        "Wave Out",
        "What U Hear",
        "Rec. Playback"
    };
    int num_keywords = sizeof(loopback_keywords) / sizeof(loopback_keywords[0]);

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

    return -1;
}

// Initialize audio stream context
AudioStreamContext* audio_stream_init(int use_system_audio,
                                      AudioFrameCallback callback,
                                      void* user_data) {
    AudioStreamContext* ctx = (AudioStreamContext*)calloc(1, sizeof(AudioStreamContext));
    if (!ctx) return NULL;

    ctx->use_system_audio = use_system_audio;
    ctx->callback = callback;
    ctx->user_data = user_data;
    ctx->sample_rate = MELSPEC_SR;
    ctx->running = 0;
    ctx->frame_counter = 0;

    // Initialize circular buffer
    int buffer_size = (int)(MELSPEC_SR * BUFFER_DURATION);
    circular_buffer_init(&ctx->buffer, buffer_size);

    // Pre-allocate audio conversion buffer
    ctx->audio_conversion_buffer_size = 2048;
    ctx->audio_conversion_buffer = (float*)malloc(ctx->audio_conversion_buffer_size * sizeof(float));
    if (!ctx->audio_conversion_buffer) {
        circular_buffer_free(&ctx->buffer);
        free(ctx);
        return NULL;
    }

    return ctx;
}

// Start the audio stream
int audio_stream_start(AudioStreamContext* ctx) {
    if (!ctx) return -1;

    ctx->soundio = soundio_create();
    if (!ctx->soundio) {
        fprintf(stderr, "Error: Out of memory\n");
        return -1;
    }

    int err = soundio_connect(ctx->soundio);
    if (err) {
        fprintf(stderr, "Error connecting: %s\n", soundio_strerror(err));
        soundio_destroy(ctx->soundio);
        ctx->soundio = NULL;
        return -1;
    }

    soundio_flush_events(ctx->soundio);

    int device_index;

    if (ctx->use_system_audio) {
        device_index = find_loopback_device(ctx->soundio);
        if (device_index < 0) {
            fprintf(stderr, "Error: No system audio loopback device found\n");
            soundio_destroy(ctx->soundio);
            ctx->soundio = NULL;
            return -1;
        }
    } else {
        device_index = soundio_default_input_device_index(ctx->soundio);
        if (device_index < 0) {
            fprintf(stderr, "Error: No input device found\n");
            soundio_destroy(ctx->soundio);
            ctx->soundio = NULL;
            return -1;
        }
    }

    ctx->device = soundio_get_input_device(ctx->soundio, device_index);
    if (!ctx->device) {
        fprintf(stderr, "Error: Out of memory\n");
        soundio_destroy(ctx->soundio);
        ctx->soundio = NULL;
        return -1;
    }

    printf("Using input device: %s\n", ctx->device->name);

    ctx->instream = soundio_instream_create(ctx->device);
    if (!ctx->instream) {
        fprintf(stderr, "Error: Out of memory\n");
        soundio_device_unref(ctx->device);
        soundio_destroy(ctx->soundio);
        ctx->device = NULL;
        ctx->soundio = NULL;
        return -1;
    }

    ctx->instream->format = SoundIoFormatFloat32NE;
    ctx->instream->sample_rate = MELSPEC_SR;
    ctx->instream->read_callback = read_callback;
    ctx->instream->overflow_callback = overflow_callback;
    ctx->instream->software_latency = 0.01;
    ctx->instream->userdata = ctx;

    if ((err = soundio_instream_open(ctx->instream))) {
        fprintf(stderr, "Error opening stream: %s\n", soundio_strerror(err));
        soundio_instream_destroy(ctx->instream);
        soundio_device_unref(ctx->device);
        soundio_destroy(ctx->soundio);
        ctx->instream = NULL;
        ctx->device = NULL;
        ctx->soundio = NULL;
        return -1;
    }

    ctx->running = 1;

    // Start processing thread
    pthread_create(&ctx->processing_thread, NULL, streaming_spectrogram_thread, ctx);

    if ((err = soundio_instream_start(ctx->instream))) {
        fprintf(stderr, "Error starting stream: %s\n", soundio_strerror(err));
        ctx->running = 0;
        pthread_join(ctx->processing_thread, NULL);
        soundio_instream_destroy(ctx->instream);
        soundio_device_unref(ctx->device);
        soundio_destroy(ctx->soundio);
        ctx->instream = NULL;
        ctx->device = NULL;
        ctx->soundio = NULL;
        return -1;
    }

    printf("Recording from microphone...\n");
    return 0;
}

// Stop the audio stream
void audio_stream_stop(AudioStreamContext* ctx) {
    if (!ctx) return;

    ctx->running = 0;
    if (ctx->processing_thread) {
        pthread_join(ctx->processing_thread, NULL);
    }

    if (ctx->instream) {
        soundio_instream_destroy(ctx->instream);
        ctx->instream = NULL;
    }
    if (ctx->device) {
        soundio_device_unref(ctx->device);
        ctx->device = NULL;
    }
    if (ctx->soundio) {
        soundio_destroy(ctx->soundio);
        ctx->soundio = NULL;
    }
}

// Check if audio stream is running
int audio_stream_is_running(AudioStreamContext* ctx) {
    return ctx ? ctx->running : 0;
}

// Free audio stream context
void audio_stream_free(AudioStreamContext* ctx) {
    if (!ctx) return;

    audio_stream_stop(ctx);
    circular_buffer_free(&ctx->buffer);
    free(ctx->audio_conversion_buffer);
    free(ctx);
}
