#include "melspec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

// Internal context structure
struct MelSpectrogramContext {
    int n_fft;
    int n_mels;
    int sample_rate;

    // Mel filterbank [n_mels][n_fft/2 + 1]
    float** filterbank;

    // FFTW plan and buffers
    float* fft_input;
    fftwf_complex* fft_output;
    fftwf_plan plan;

    // Working buffers
    float* power_spec;
};

// Mel scale conversion functions
float melspec_hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melspec_mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Create mel filterbank - matching librosa's implementation
static void create_mel_filterbank(float** filterbank, int n_mels, int n_fft, int sr) {
    float fmin = 0.0f;
    float fmax = sr / 2.0f;

    // Create mel points
    float min_mel = melspec_hz_to_mel(fmin);
    float max_mel = melspec_hz_to_mel(fmax);

    float* mels = (float*)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        mels[i] = min_mel + (max_mel - min_mel) * i / (n_mels + 1);
    }

    // Convert back to Hz
    float* freqs = (float*)malloc((n_mels + 2) * sizeof(float));
    for (int i = 0; i < n_mels + 2; i++) {
        freqs[i] = melspec_mel_to_hz(mels[i]);
    }

    // FFT bin frequencies
    int n_freqs = n_fft / 2 + 1;
    float* fft_freqs = (float*)malloc(n_freqs * sizeof(float));
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
void melspec_apply_hanning_window(float* frame, int n_fft) {
    for (int i = 0; i < n_fft; i++) {
        float window = 0.5f - 0.5f * cosf(2.0f * M_PI * i / n_fft);
        frame[i] *= window;
    }
}

// Initialize mel spectrogram context
MelSpectrogramContext* melspec_init(int n_fft, int n_mels, int sample_rate) {
    MelSpectrogramContext* ctx = (MelSpectrogramContext*)calloc(1, sizeof(MelSpectrogramContext));
    if (!ctx) return NULL;

    ctx->n_fft = n_fft;
    ctx->n_mels = n_mels;
    ctx->sample_rate = sample_rate;

    // Allocate mel filterbank
    ctx->filterbank = (float**)malloc(n_mels * sizeof(float*));
    if (!ctx->filterbank) {
        free(ctx);
        return NULL;
    }

    for (int i = 0; i < n_mels; i++) {
        ctx->filterbank[i] = (float*)calloc(n_fft / 2 + 1, sizeof(float));
        if (!ctx->filterbank[i]) {
            for (int j = 0; j < i; j++) free(ctx->filterbank[j]);
            free(ctx->filterbank);
            free(ctx);
            return NULL;
        }
    }

    create_mel_filterbank(ctx->filterbank, n_mels, n_fft, sample_rate);

    // FFTW setup
    ctx->fft_input = (float*)fftwf_malloc(n_fft * sizeof(float));
    ctx->fft_output = (fftwf_complex*)fftwf_malloc((n_fft / 2 + 1) * sizeof(fftwf_complex));

    if (!ctx->fft_input || !ctx->fft_output) {
        melspec_free(ctx);
        return NULL;
    }

    ctx->plan = fftwf_plan_dft_r2c_1d(n_fft, ctx->fft_input, ctx->fft_output, FFTW_MEASURE);
    if (!ctx->plan) {
        melspec_free(ctx);
        return NULL;
    }

    // Allocate power spectrum buffer
    ctx->power_spec = (float*)malloc((n_fft / 2 + 1) * sizeof(float));
    if (!ctx->power_spec) {
        melspec_free(ctx);
        return NULL;
    }

    return ctx;
}

// Free mel spectrogram context
void melspec_free(MelSpectrogramContext* ctx) {
    if (!ctx) return;

    if (ctx->filterbank) {
        for (int i = 0; i < ctx->n_mels; i++) {
            free(ctx->filterbank[i]);
        }
        free(ctx->filterbank);
    }

    if (ctx->plan) fftwf_destroy_plan(ctx->plan);
    if (ctx->fft_input) fftwf_free(ctx->fft_input);
    if (ctx->fft_output) fftwf_free(ctx->fft_output);
    if (ctx->power_spec) free(ctx->power_spec);

    free(ctx);
}

// Process a single frame of audio data and add to rolling buffer
void melspec_process_frame(MelSpectrogramContext* ctx,
                          const float* audio_frame,
                          float* mel_frame_out) {
    if (!ctx || !audio_frame || !mel_frame_out) return;

    // Copy audio to FFT input buffer
    memcpy(ctx->fft_input, audio_frame, ctx->n_fft * sizeof(float));

    // Execute FFT
    fftwf_execute(ctx->plan);

    // Compute power spectrum
    for (int i = 0; i < ctx->n_fft / 2 + 1; i++) {
        float re = ctx->fft_output[i][0];
        float im = ctx->fft_output[i][1];
        ctx->power_spec[i] = (re * re + im * im) / (ctx->n_fft * ctx->n_fft);
    }

    // Apply mel filterbank
    for (int mel = 0; mel < ctx->n_mels; mel++) {
        float mel_power = 0.0f;
        for (int i = 0; i < ctx->n_fft / 2 + 1; i++) {
            mel_power += ctx->filterbank[mel][i] * ctx->power_spec[i];
        }
        mel_frame_out[mel] = mel_power;
    }
}

// Rolling buffer functions
void melspec_rolling_buffer_init(MelRollingBuffer* mrb, int max_frames, int n_mels) {
    mrb->max_frames = max_frames;
    mrb->current_frames = 0;
    mrb->write_pos = 0;
    mrb->frames = (float**)malloc(max_frames * sizeof(float*));
    for (int i = 0; i < max_frames; i++) {
        mrb->frames[i] = (float*)calloc(n_mels, sizeof(float));
    }
}

void melspec_rolling_buffer_free(MelRollingBuffer* mrb) {
    if (!mrb || !mrb->frames) return;

    for (int i = 0; i < mrb->max_frames; i++) {
        free(mrb->frames[i]);
    }
    free(mrb->frames);
    mrb->frames = NULL;
}

void melspec_rolling_buffer_add_frame(MelRollingBuffer* mrb,
                                     const float* mel_frame,
                                     int n_mels) {
    if (!mrb || !mel_frame) return;

    memcpy(mrb->frames[mrb->write_pos], mel_frame, n_mels * sizeof(float));
    mrb->write_pos = (mrb->write_pos + 1) % mrb->max_frames;
    if (mrb->current_frames < mrb->max_frames) {
        mrb->current_frames++;
    }
}

// Convert mel spectrogram buffer to dB scale and normalize
void melspec_to_db_normalized(const MelRollingBuffer* mrb,
                              float** db_values,
                              int output_width,
                              float* min_db_out,
                              float* max_db_out) {
    if (!mrb || !db_values || output_width <= 0) return;

    const float amin = 1e-10f;
    const float top_db = 80.0f;

    // Step 1: Find max power across all frames (for dB reference)
    float max_power = 1e-10f;
    for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
        int buffer_idx = (mrb->write_pos - output_width + frame_idx + mrb->max_frames) % mrb->max_frames;
        for (int mel = 0; mel < MELSPEC_N_MELS; mel++) {
            float power = mrb->frames[buffer_idx][mel];
            if (power > max_power) max_power = power;
        }
    }

    // Step 2: Convert to dB and find min/max for normalization
    float min_db = 0.0f, max_db = 0.0f;
    for (int frame_idx = 0; frame_idx < output_width; frame_idx++) {
        int buffer_idx = (mrb->write_pos - output_width + frame_idx + mrb->max_frames) % mrb->max_frames;
        for (int mel = 0; mel < MELSPEC_N_MELS; mel++) {
            float power = fmaxf(mrb->frames[buffer_idx][mel], amin);
            float ref = fmaxf(max_power, amin);
            float db = 10.0f * log10f(power / ref);

            // Apply top_db threshold
            if (db < -top_db) db = -top_db;

            db_values[frame_idx][mel] = db;
            if (db < min_db) min_db = db;
            if (db > max_db) max_db = db;
        }
    }

    if (min_db_out) *min_db_out = min_db;
    if (max_db_out) *max_db_out = max_db;
}
