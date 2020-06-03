#include <iostream>
#include <cassert>

#define GL_GLEXT_PROTOTYPES 1
#define GL3_PROTOTYPES      1

#define EPS 0.001
#define WIDTH 1024
#define HEIGHT 768
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
  if (storePos.x >= 1024 || storePos.y >= 768) return;

  vec2 uv = vec2(float(gl_GlobalInvocationID.x), float(gl_GlobalInvocationID.y)) / vec2(1024, 768);
  uv = uv * 2 - vec2(1);

  vec3 tangent = cross(viewdir, vec3(0,0,1));
  vec3 bitangent = cross(viewdir, tangent);

  vec3 center = eye + d * viewdir;
  vec3 point = center + uv.x * tangent - uv.y * bitangent;
  vec3 dir = normalize(point - eye);

  Ray ray;
  ray.origin = vec4(point, 100);
  ray.direction = vec4(dir, 0);

  rays[storePos.x + 1024 * storePos.y] = ray;
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
   vec4 color;
};

struct AABB {
  vec4 vmin;
  vec4 vmax;
  vec4 color;
  vec4 luminance;
};

layout(std430, binding = 1) buffer triangleBuf
{
  Triangle triangles[];
};

layout(std430, binding = 2) buffer rayBuf
{
  Ray rays[];
};

layout(std430, binding = 3) buffer aabbBuf
{
  AABB boundingBoxes[];
};


layout(location = 0) uniform float time;

ivec2 screen_size = ivec2(1024, 480);
vec3 seed;

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}



// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }



// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}



// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float rand() { float r =  random(seed); seed.z = r; return r; }


bool intersect(inout Ray ray, in Triangle triangle, out vec3 normal) {
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
      normal = cross(edge0, edge1);
      return true;
    }

    return false;
}

bool intersect(inout Ray ray, int index) {
    AABB box = boundingBoxes[index];
    float tx1 = (box.vmin.x - ray.origin.x)/ray.direction.x;
    float tx2 = (box.vmax.x - ray.origin.x)/ray.direction.x;

    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (box.vmin.y - ray.origin.y)/ray.direction.y;
    float ty2 = (box.vmax.y - ray.origin.y)/ray.direction.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (box.vmin.z - ray.origin.z)/ray.direction.z;
    float tz2 = (box.vmax.z - ray.origin.z)/ray.direction.z;

    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    if (tmax >= tmin && tmax >= 0 && tmin < ray.origin.w)
    {
        ray.origin.w = tmin;
        return true;
    }

    return false;
}

vec3 sampleHalfDome(vec3 w) {
   // create 2 random numbers
   float r1 = 2 * 3.1415927 * rand(); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
   float r2 = rand();    // pick random number for elevation
   float r2s = sqrt(r2);

   vec3 u = normalize(cross((abs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
   vec3 v = cross(w,u);
   return normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
}


vec3 sampleHalfDome2(mat3 TBN)
{
    float u1 = rand() * 2 - 1;
    float u2 = rand() * 2 - 1;
    float r = sqrt(u1);
    float theta = 2 * 3.1415 * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return TBN * vec3(x, y, sqrt(max(0.0f, 1 - u1)));
}

vec3 sampleLightDir(vec3 o)
{
    int id = int(rand() * 6);
    vec3 origin = boundingBoxes[id].vmin.xyz;
    vec3 dir = origin - boundingBoxes[id].vmax.xyz;
    return normalize(vec3(origin.x + rand() * dir.x, origin.y + rand() * dir.y, origin.z + rand() * dir.z) - o);
}

vec3 recoverNormalFromAABB(int index, vec3 point)
{
      AABB box = boundingBoxes[index];
      vec3 center = (box.vmin.xyz + box.vmax.xyz) * 0.5;
      vec3 vmin_point = abs(point - box.vmin.xyz);
      vec3 vmax_point = abs(point - box.vmax.xyz);
      if (min(vmin_point.x, vmax_point.x) < 0.0005) return vec3(1 * sign(point.x - center.x),0,0);
      if (min(vmin_point.y, vmax_point.y) < 0.0005) return vec3(0,1 * sign(point.y - center.y),0);
      if (min(vmin_point.z, vmax_point.z) < 0.0005) return vec3(0,0,1 * sign(point.z - center.z));
      return vec3(0);
}

int queryScene(inout Ray ray)
{
  int c = -1;
  for(int i=0; i<6; i++)
  {
      if (intersect(ray, i)) c = i;
  }
  return c;
}

vec4 sampleRay(in Ray ray) {
  vec4 accucolor = vec4(0);
  vec4 mask = vec4(1);
  for(int i=0; i<4; i++) {
    int collider = queryScene(ray);

    // miss! return black
    if (collider == -1) return accucolor + vec4(0.4, 0.4, 0.6, 0.6);

    vec3 point = ray.origin.xyz + ray.direction.xyz * ray.origin.w;
    vec3 normal = recoverNormalFromAABB(collider, point);
    vec3 rand_vec = normalize(vec3(rand()*2-1, rand()*2-1, rand()*2-1));
    vec3 tangent = normalize(cross(normal, rand_vec));
    vec3 bitangent = normalize(cross(normal, tangent));
    mat3 TBN = mat3(bitangent, tangent, normal);

    accucolor += mask * boundingBoxes[collider].luminance;

    // Construct new ray
    ray.origin = vec4(ray.origin.xyz + ray.direction.xyz * (ray.origin.w - 0.001), 100);
    ray.direction.xyz = sampleHalfDome(normal);

    mask *= boundingBoxes[collider].color;
    mask *= dot(ray.direction.xyz, normal);
    mask *= 1.5;
  }

  return accucolor;
}

void main()
{
  seed = vec3(gl_GlobalInvocationID.xy, time) * time;
  ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);

  Ray ray = rays[storePos.x + screen_size.x * storePos.y];
  ray.origin.x += rand() * 2.0 / screen_size.x;
  ray.origin.z += rand() * 2.0 / screen_size.y;
  vec4 color = sampleRay(ray);

  vec4 oldcolor = imageLoad(destTex, storePos);
  float a = 0.999;
  imageStore(destTex, storePos, color * (1-a) + a * oldcolor);
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
    Vector4 color;
};

struct AABB
{
    Vector4 vmin;
    Vector4 vmax;
    Vector4 color;
    Vector4 luminance;
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
  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Pathtracer", NULL, NULL);
  if (!window)
  {
    return -3;
  }
  glfwMakeContextCurrent(window);
  glDisable(GL_DEPTH_TEST);
  glfwSwapInterval(0);

  auto cs_gen_rays = GenerateProgram(CompileShader(GL_COMPUTE_SHADER, cs_genrays_shader));

  auto cs = CompileShader(GL_COMPUTE_SHADER, cs_pathtrace_shader);
  auto cs_program = GenerateProgram(cs);

  Triangle triangles[6] = {
          // Left wall
          { { -1, 0, -1, 0 }, { -1, 0, 1, 0}, { -1, 4, -1, 0 }, {0, 0, 1, 1} },
          { { -1, 0, 1, 0 }, { -1, 4, -1, 0}, { -1, 4, 1, 0 }, {0, 0, 1, 1} },

          // Right wall
          { { 1, 0, -1, 0 }, { 1, 0, 1, 0}, { 1, 4, -1, 0 }, {0, 0, 1, 1} },
          { { 1, 0, 1, 0 }, { 1, 4, -1, 0}, { 1, 4, 1, 0 }, {0, 0, 1, 1} },

          // Floor
          { { -1, 0, -1, 0}, {-1, 4, -1, 0}, { 1, 4, -1, 0}, {1,1,1,1} },
          { { 1, 0, -1, 0}, {1, 4, -1, 0}, { -1, 0, -1, 0}, {1,1,1,1} },
  };

  AABB boundingBoxes[6] = {
          { { -1, 0, -1, 0}, { 1, 4, -1.2, 0}, { 1, 1, 1, 1}, { 0, 0, 0, 0 }},
          { { -0.4, 1.5, -0.2, 0}, { 0.4, 2.5, -0.7, 0}, { 0.8, 0.3, 0.4, 1}, { 0.0, 0.0, 0.0, 0 }},
          { { -0.3, 0, 1, 0}, { 0.3, 2.3, 1.2, 0}, { 0, 0, 1, 1}, { 2, 2, 2, 0 }},
          { { -1, 0, -1, 0}, { -1.1, 4, 1, 0}, { 1, 0, 0, 1}, { 0, 0, 0.0, 0 }},
          { { 1, 0, -1, 0}, { 1.1, 4, 1, 0}, { 1, 0, 1, 1}, { 0, 0, 0, 0 }},
          { { -5, 3.3, -5, 0}, { 5, 3.4, 1, 0}, { 0, 1, 1, 1}, { 0, 0, 0, 0 }},
  };

  GLuint triangle_buf;
  glGenBuffers(1, &triangle_buf);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, triangle_buf);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(triangles), triangles, GL_STATIC_DRAW);

  GLuint boundingBox_buf;
  glGenBuffers(1, &boundingBox_buf);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, boundingBox_buf);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(boundingBoxes), boundingBoxes, GL_STATIC_DRAW);

  GLuint ray_buf;
  glGenBuffers(1, &ray_buf);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ray_buf);
  glBufferData(GL_SHADER_STORAGE_BUFFER, WIDTH * HEIGHT * sizeof(Ray), nullptr, GL_DYNAMIC_DRAW);

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
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
  float d = 1.4;
  while (!glfwWindowShouldClose(window))
  {
  //  glBindBuffer(GL_SHADER_STORAGE_BUFFER, boundingBox_buf);
 //   AABB* boxes = (AABB*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_WRITE);
   // boxes[2].vmin.z -= 0.001f;
    //boxes[2].vmax.z -= 0.001f;
//    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glUseProgram(cs_gen_rays);
    glUniform1f(0, glfwGetTime());
    glUniform3f(1, eye.x, eye.y, eye.z);
    glUniform3f(2, viewdir.x, viewdir.y, viewdir.z);
    glUniform1f(3, d);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_buf);
    glDispatchCompute(WIDTH / 16, HEIGHT / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(cs_program);
    glUniform1f(0, glfwGetTime());
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, triangle_buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ray_buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, boundingBox_buf);
    glDispatchCompute(WIDTH / 16, HEIGHT / 16, 1);
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
