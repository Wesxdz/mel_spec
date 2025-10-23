#ifndef MELSPEC_H
#define MELSPEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration constants
#define MELSPEC_N_FFT 1024
#define MELSPEC_HOP_LENGTH 256
#define MELSPEC_N_MELS 128
#define MELSPEC_SR 22050

// Rolling buffer for mel spectrogram frames
typedef struct {
    float **frames;           // 2D array: [frame_index][mel_bin]
    int max_frames;          // Maximum number of frames to store
    int current_frames;      // Current number of frames stored
    int write_pos;           // Circular buffer write position
} MelRollingBuffer;

// Mel spectrogram context (opaque handle)
typedef struct MelSpectrogramContext MelSpectrogramContext;

// Initialize mel spectrogram context
// Returns NULL on failure
MelSpectrogramContext* melspec_init(int n_fft, int n_mels, int sample_rate);

// Free mel spectrogram context
void melspec_free(MelSpectrogramContext* ctx);

// Process a single frame of audio data and add to rolling buffer
// audio_frame: N_FFT samples of audio data
// mel_frame_out: Output buffer for N_MELS mel coefficients (caller-allocated)
void melspec_process_frame(MelSpectrogramContext* ctx,
                          const float* audio_frame,
                          float* mel_frame_out);

// Rolling buffer functions
void melspec_rolling_buffer_init(MelRollingBuffer* mrb, int max_frames, int n_mels);
void melspec_rolling_buffer_free(MelRollingBuffer* mrb);
void melspec_rolling_buffer_add_frame(MelRollingBuffer* mrb,
                                     const float* mel_frame,
                                     int n_mels);

// Apply Hanning window to audio frame (in-place)
void melspec_apply_hanning_window(float* frame, int n_fft);

// Mel scale conversion functions
float melspec_hz_to_mel(float hz);
float melspec_mel_to_hz(float mel);

// Convert mel spectrogram buffer to dB scale and normalize
// Input: mrb - rolling buffer with power values
// Output: db_values - 2D array [frame_idx][mel_bin] with normalized dB values [0, 1]
// Returns: min_db, max_db via output pointers
void melspec_to_db_normalized(const MelRollingBuffer* mrb,
                              float** db_values,
                              int output_width,
                              float* min_db_out,
                              float* max_db_out);

#ifdef __cplusplus
}
#endif

#endif // MELSPEC_H
