#ifndef AUDIO_STREAM_H
#define AUDIO_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque audio stream context
typedef struct AudioStreamContext AudioStreamContext;

// Callback function for when new spectrogram image data is ready
// image_data: RGB image data (width * height * 3 bytes)
// width: Image width in pixels
// height: Image height in pixels
// user_data: User-provided context pointer
typedef void (*AudioFrameCallback)(unsigned char* image_data, int width, int height, void* user_data);

// Initialize audio stream context
// use_system_audio: If 1, use system audio loopback; if 0, use microphone
// callback: Function to call when new image data is ready
// user_data: User context pointer passed to callback
// Returns: AudioStreamContext pointer on success, NULL on failure
AudioStreamContext* audio_stream_init(int use_system_audio,
                                      AudioFrameCallback callback,
                                      void* user_data);

// Start the audio stream (begins capturing and processing)
// Returns: 0 on success, non-zero on error
int audio_stream_start(AudioStreamContext* ctx);

// Stop the audio stream
void audio_stream_stop(AudioStreamContext* ctx);

// Check if audio stream is running
// Returns: 1 if running, 0 if stopped
int audio_stream_is_running(AudioStreamContext* ctx);

// Free audio stream context and all resources
void audio_stream_free(AudioStreamContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_STREAM_H
