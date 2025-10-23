#include "audio_stream.h"
#include "melspec.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>

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

    // pipewire objects
    struct pw_thread_loop *loop;
    struct pw_stream *stream;
    struct spa_hook stream_listener;
    int n_channels;

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

// pipewire stream process callback
static void on_stream_process(void *userdata) {
    AudioStreamContext *ctx = (AudioStreamContext*)userdata;
    struct pw_buffer *b;
    struct spa_buffer *buf;
    float *samples;
    uint32_t n_samples;

    if ((b = pw_stream_dequeue_buffer(ctx->stream)) == NULL) {
        fprintf(stderr, "out of buffers\n");
        return;
    }

    buf = b->buffer;
    if (buf->datas[0].data == NULL || buf->datas[0].chunk->size == 0) {
        pw_stream_queue_buffer(ctx->stream, b);
        return;
    }

    samples = buf->datas[0].data;
    n_samples = buf->datas[0].chunk->size / sizeof(float);

    // Convert multi-channel to mono if needed
    int channels = ctx->n_channels;
    if (channels <= 0) channels = 2; // Default to stereo

    int frames = n_samples / channels;

    if (frames > ctx->audio_conversion_buffer_size) {
        frames = ctx->audio_conversion_buffer_size;
    }

    for (int frame = 0; frame < frames; frame++) {
        float sample = 0.0f;

        // Average all channels to mono
        for (int ch = 0; ch < channels; ch++) {
            sample += samples[frame * channels + ch];
        }
        sample /= channels;

        ctx->audio_conversion_buffer[frame] = sample;
    }

    circular_buffer_write(&ctx->buffer, ctx->audio_conversion_buffer, frames);

    pw_stream_queue_buffer(ctx->stream, b);
}

// pipewire stream param changed callback
static void on_stream_param_changed(void *userdata, uint32_t id, const struct spa_pod *param) {
    AudioStreamContext *ctx = (AudioStreamContext*)userdata;

    if (param == NULL || id != SPA_PARAM_Format)
        return;

    struct spa_audio_info_raw info = { 0 };
    if (spa_format_audio_raw_parse(param, &info) < 0)
        return;

    ctx->n_channels = info.channels;
    printf("Stream format: %d channels, %d Hz\n", info.channels, info.rate);
}

// pipewire stream state changed callback
static void on_stream_state_changed(void *userdata, enum pw_stream_state old,
                                    enum pw_stream_state state, const char *error) {
    AudioStreamContext *ctx = (AudioStreamContext*)userdata;
    printf("Stream state changed: %s -> %s\n",
           pw_stream_state_as_string(old),
           pw_stream_state_as_string(state));

    if (state == PW_STREAM_STATE_ERROR) {
        fprintf(stderr, "Stream error: %s\n", error);
        ctx->running = 0;
    }
}

// pipewire stream events
static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_stream_process,
    .state_changed = on_stream_state_changed,
    .param_changed = on_stream_param_changed,
};

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
    ctx->n_channels = 2;  // Default to stereo, will be updated by param_changed

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

    // Initialize pipewire
    pw_init(NULL, NULL);

    // Create thread loop
    ctx->loop = pw_thread_loop_new("audio-capture", NULL);
    if (!ctx->loop) {
        fprintf(stderr, "Failed to create pipewire thread loop\n");
        return -1;
    }

    // Create stream
    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Audio",
        PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_ROLE, "Music",
        NULL
    );

    // Set target for system audio or microphone
    if (ctx->use_system_audio) {
        // Connect to system audio monitor/loopback
        // PW_KEY_STREAM_CAPTURE_SINK tells pipewire to capture from the sink monitor
        pw_properties_set(props, PW_KEY_STREAM_CAPTURE_SINK, "true");
        printf("Requesting system audio loopback...\n");
    } else {
        // Connect to default microphone - no special properties needed
        printf("Requesting microphone input...\n");
    }

    ctx->stream = pw_stream_new_simple(
        pw_thread_loop_get_loop(ctx->loop),
        "audio-capture",
        props,
        &stream_events,
        ctx
    );

    if (!ctx->stream) {
        fprintf(stderr, "Failed to create pipewire stream\n");
        pw_thread_loop_destroy(ctx->loop);
        ctx->loop = NULL;
        return -1;
    }

    // Configure audio format
    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    const struct spa_pod *params[1];
    struct spa_audio_info_raw audio_info = {
        .format = SPA_AUDIO_FORMAT_F32,
        .rate = MELSPEC_SR,
        .channels = 2,  // Request stereo, we'll convert to mono
    };

    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &audio_info);

    // Connect stream
    if (pw_stream_connect(ctx->stream,
                         PW_DIRECTION_INPUT,
                         PW_ID_ANY,
                         PW_STREAM_FLAG_AUTOCONNECT |
                         PW_STREAM_FLAG_MAP_BUFFERS |
                         PW_STREAM_FLAG_RT_PROCESS,
                         params, 1) < 0) {
        fprintf(stderr, "Failed to connect pipewire stream\n");
        pw_stream_destroy(ctx->stream);
        pw_thread_loop_destroy(ctx->loop);
        ctx->stream = NULL;
        ctx->loop = NULL;
        return -1;
    }

    // Start the thread loop
    if (pw_thread_loop_start(ctx->loop) < 0) {
        fprintf(stderr, "Failed to start pipewire thread loop\n");
        pw_stream_destroy(ctx->stream);
        pw_thread_loop_destroy(ctx->loop);
        ctx->stream = NULL;
        ctx->loop = NULL;
        return -1;
    }

    ctx->running = 1;

    // Start processing thread
    pthread_create(&ctx->processing_thread, NULL, streaming_spectrogram_thread, ctx);

    if (ctx->use_system_audio) {
        printf("Capturing system audio...\n");
    } else {
        printf("Recording from microphone...\n");
    }

    return 0;
}

// Stop the audio stream
void audio_stream_stop(AudioStreamContext* ctx) {
    if (!ctx) return;

    ctx->running = 0;
    if (ctx->processing_thread) {
        pthread_join(ctx->processing_thread, NULL);
    }

    if (ctx->loop) {
        pw_thread_loop_stop(ctx->loop);
    }

    if (ctx->stream) {
        pw_stream_destroy(ctx->stream);
        ctx->stream = NULL;
    }

    if (ctx->loop) {
        pw_thread_loop_destroy(ctx->loop);
        ctx->loop = NULL;
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
