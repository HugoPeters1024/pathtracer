#include <iostream>
#include <cassert>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1

#define EPS 0.001
#include <GLFW/glfw3.h>
#include "gl_utils/vec.h"
#include "gl_utils/shader_utils.h"
#include "gl_utils/gl_debug.h"

static const char* quad_shader_vs = R"(
#version 450

layout(location = 0) in vec2 vPos;

out vec2 uv;

void main()
{
  gl_Position = vec4(vPos, 0, 1);
  uv = (vPos + vec2(1)) * 0.5f;
}
)";

static const char* quad_shader_fs = R"(
#version 450

layout(location = 0) uniform sampler2D tex;
in vec2 uv;
out vec4 color;

void main()
{
  color = texture(tex, uv);
}
)";

static const char* cs_genrays_shader = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(location = 0) uniform float time;
layout(location = 1) uniform vec3 eye;
layout(location = 2) uniform vec3 viewdir;
layout(location = 3) uniform float d;

struct Ray {
   vec4 origin;
   vec4 direction;
};

layout(std430, binding = 2) buffer rayBuf
{
  Ray rays[];
};

void main()
{
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
  if (storePos.x >= 640 || storePos.y >= 480) return;

  vec2 uv = vec2(float(gl_GlobalInvocationID.x), float(gl_GlobalInvocationID.y)) / vec2(640, 480);

  vec3 tangent = cross(viewdir, vec3(0,0,1));
  vec3 bitangent = cross(viewdir, tangent);

  vec3 center = eye + d * viewdir;
  vec3 point = center + uv.x * tangent - uv.y * bitangent;
  vec3 dir = normalize(point - eye);

  Ray ray;
  ray.origin = vec4(point, 100);
  ray.direction = vec4(dir, 0);

  rays[storePos.x + 640 * storePos.y] = ray;
}
)";

static const char* cs_pathtrace_shader = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) uniform image2D destTex;

struct Ray {
   vec4 origin;
   vec4 direction;
};

struct Triangle {
   vec4 v0;
   vec4 v1;
   vec4 v2;
};

layout(std430, binding = 1) buffer triangleBuf
{
  Triangle triangles[];
};

layout(std430, binding = 2) buffer rayBuf
{
  Ray rays[];
};


layout(location = 0) uniform float time;
ivec2 screen_size = ivec2(640, 480);


bool intersect(in Ray ray, in Triangle triangle) {
    vec3 ray_origin = ray.origin.xyz;
    vec3 ray_direction = ray.direction.xyz;
    vec3 v0 = triangle.v0.xyz;
    vec3 v1 = triangle.v1.xyz;
    vec3 v2 = triangle.v2.xyz;

    // compute plane's normal
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    // no need to normalize
    vec3 N = cross(v0v1, v0v2); // N

    // Step 1: finding P
    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N, ray_direction);
    if (abs(NdotRayDirection) < 0.001) // almost 0
      return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = -dot(N, v0);

    // compute t (equation 3)
    float t = -(dot(N, ray_origin) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    vec3 P = ray_origin + t * ray_direction;

    // Step 2: inside-outside test
    vec3 C; // vector perpendicular to triangle's plane

    // edge 0
    vec3 edge0 = v1 - v0;
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0) return false; // P is on the right side


    // edge 1
    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    if (dot(N, C) < 0)  return false; // P is on the right side

    // edge 2
    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0) return false; // P is on the right side;

    // behind test already done
    if (t < ray.origin.w) {
      ray.origin.w = t;
      return true;
    }

    return false;
}

void main()
{
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
  imageStore(destTex, storePos, vec4(0,0,0,1));
  Ray ray = rays[storePos.x + screen_size.x * storePos.y];

  for(int t=0; t<1; t++)
  {
    Triangle triangle = triangles[t];

    if (intersect(ray, triangle))
      imageStore(destTex, storePos, vec4(0,1,0,1));
  }
}
)";

void error_callback(int error, const char* description)
{
  fprintf(stderr, "Error: %s\n", description);
}

struct Triangle
{
    Vector4 v0;
    Vector4 v1;
    Vector4 v2;
};

struct Ray
{
    Vector4 origin;
    Vector4 direction;
};

int main() {
  if (!glfwInit())
  {
    return -2;
  }

  glfwSetErrorCallback(error_callback);
  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback(GLDEBUGPROC(gl_debug_output), nullptr);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  GLFWwindow* window = glfwCreateWindow(640, 480, "Pathtracer", NULL, NULL);
  if (!window)
  {
    return -3;
  }
  glfwMakeContextCurrent(window);
  glDisable(GL_DEPTH_TEST);

  auto cs_gen_rays = GenerateProgram(CompileShader(GL_COMPUTE_SHADER, cs_genrays_shader));

  auto cs = CompileShader(GL_COMPUTE_SHADER, cs_pathtrace_shader);
  auto cs_program = GenerateProgram(cs);

  Triangle triangles[1] = {
          { { 0.2, 1, 0.2, 0 }, { 0.8, 1, 0.2, 0}, { 0.5, 1, 0.8, 0 } },
  };

  GLuint triangle_buf;
  glGenBuffers(1, &triangle_buf);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, triangle_buf);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(triangles), triangles, GL_STATIC_DRAW);

  GLuint ray_buf;
  glGenBuffers(1, &ray_buf);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ray_buf);
  glBufferData(GL_SHADER_STORAGE_BUFFER, 640 * 480 * sizeof(Ray), nullptr, GL_DYNAMIC_DRAW);

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 640, 480, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  auto quad_vs = CompileShader(GL_VERTEX_SHADER, quad_shader_vs);
  auto quad_fs = CompileShader(GL_FRAGMENT_SHADER, quad_shader_fs);
  auto quad_program = GenerateProgram(quad_vs, quad_fs);
  glUseProgram(quad_program);

  float quad_data[12] = {
          -1.0, -1.0,
          1.0, -1.0,
          -1.0, 1.0,

          1.0, 1.0,
          -1.0, 1.0,
          1.0, -1.0,
  };

  GLuint quad_vao, quad_vbo;
  glGenVertexArrays(1, &quad_vao);
  glGenBuffers(1, &quad_vbo);

  glBindVertexArray(quad_vao);
  glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad_data), quad_data, GL_STATIC_DRAW);

  glEnableVertexArrayAttrib(quad_vao, 0);
  glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, nullptr);

  Vector3 eye(0,0,0);
  Vector3 viewdir(0,1,0);
  float d = 1;
  while (!glfwWindowShouldClose(window))
  {
    glUseProgram(cs_gen_rays);
    glUniform1f(0, glfwGetTime());
    glUniform3f(1, eye.x, eye.y, eye.z);
    glUniform3f(2, viewdir.x, viewdir.y, viewdir.z);
    glUniform1f(3, d);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_buf);
    glDispatchCompute(640 / 16, 480 / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(cs_program);
    glUniform1f(0, glfwGetTime());
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, triangle_buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_buf);
    glDispatchCompute(640 / 16, 480 / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(quad_program);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(quad_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwPollEvents();
    glfwSwapBuffers(window);
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
