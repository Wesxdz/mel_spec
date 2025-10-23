#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "audio_stream.h"

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
    AudioStreamContext* audio_ctx; // Audio stream context
} GLContext;

static GLContext g_gl_context = {0};

// Forward declarations
static void queue_fbo_texture_update(GLContext* ctx, unsigned char* image_data, int width, int height);

// Audio frame callback - called from audio_stream library with new image data
static void audio_frame_callback(unsigned char* image_data, int width, int height, void* user_data) {
    GLContext* ctx = (GLContext*)user_data;
    queue_fbo_texture_update(ctx, image_data, width, height);
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s [--mic|--system-audio]\n", argv[0]);
        printf("  --mic              Use microphone input (default)\n");
        printf("  --system-audio     Use system audio/loopback\n");
        return 1;
    }

    // Check for streaming flags
    int use_system_audio = 0;
    if (strcmp(argv[1], "--system-audio") == 0) {
        use_system_audio = 1;
    }

    // Calculate rolling window dimensions for FBO texture
    int rolling_frames = (int)((5.0f * 22050) / 256);  // 5 second rolling window
    int tex_width = rolling_frames;
    int tex_height = 128;  // N_MELS

    // Initialize OpenGL context
    if (init_gl_context(&g_gl_context, tex_width, tex_height) != 0) {
        fprintf(stderr, "Failed to initialize OpenGL context\n");
        return 1;
    }

    // Initialize audio stream with callback
    g_gl_context.audio_ctx = audio_stream_init(use_system_audio, audio_frame_callback, &g_gl_context);
    if (!g_gl_context.audio_ctx) {
        fprintf(stderr, "Failed to initialize audio stream\n");
        glfwTerminate();
        return 1;
    }

    // Start audio stream
    if (audio_stream_start(g_gl_context.audio_ctx) != 0) {
        fprintf(stderr, "Failed to start audio stream\n");
        audio_stream_free(g_gl_context.audio_ctx);
        glfwTerminate();
        return 1;
    }

    printf("Press ESC or close window to exit\n");

    // Main render loop
    while (!glfwWindowShouldClose(g_gl_context.window)) {
        // Handle GLFW events
        glfwPollEvents();

        // Check for ESC key
        if (glfwGetKey(g_gl_context.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(g_gl_context.window, 1);
        }

        // Render FBO texture to screen
        render_fbo_to_screen(&g_gl_context);
    }

    // Cleanup
    audio_stream_stop(g_gl_context.audio_ctx);
    audio_stream_free(g_gl_context.audio_ctx);
    glfwTerminate();

    return 0;
}
