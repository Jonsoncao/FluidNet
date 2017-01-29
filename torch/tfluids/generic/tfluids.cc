// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TH_GENERIC_FILE
  #define TH_GENERIC_FILE "generic/tfluids.cc"
#else

#include <assert.h>
#include <memory>

#include "generic/vec3.cc"
#include "generic/grid.cc"

#ifdef BUILD_GL_FUNCS
  #if defined (__APPLE__) || defined (OSX)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
  #else
    #include <GL/gl.h>
  #endif

  #ifndef GLUT_API_VERSION
    #if defined(macintosh) || defined(__APPLE__) || defined(OSX)
      #include <GLUT/glut.h>
    #elif defined (__linux__) || defined (UNIX) || defined(WIN32) || defined(_WIN32)
      #include "GL/glut.h"
    #endif
  #endif
#endif

// *****************************************************************************
// advectScalar
// *****************************************************************************

inline real SemiLagrange(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, tfluids_(RealGrid)& src, 
    real dt, bool is_levelset, int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  const real p5 = static_cast<real>(0.5);
  tfluids_(vec3) pos =
      (tfluids_(vec3)((real)i + p5, (real)j + p5, (real)k + p5) -
       vel.getCentered(i, j, k, b) * dt);
  return src.getInterpolatedHi(pos, order_space, b);
}

inline real MacCormackCorrect(
    tfluids_(FlagGrid)& flags, const tfluids_(RealGrid)& old,
    const tfluids_(RealGrid)& fwd, const tfluids_(RealGrid)& bwd,
    const real strength, bool is_levelset, int32_t i, int32_t j, int32_t k,
    int32_t b) {
  real dst = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    dst += strength * 0.5 * (old(i, j, k, b) - bwd(i, j, k, b));
  }
  return dst;
}

inline void getMinMax(real& minv, real& maxv, const real& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
}

inline real clamp(const real val, const real min, const real max) {
  return std::min<real>(max, std::max<real>(min, val));
}

inline real doClampComponent(
    const Int3& gridSize, real dst, const tfluids_(RealGrid)& orig, real fwd,
    const tfluids_(vec3)& pos, const tfluids_(vec3)& vel, int32_t b) { 
  real minv = std::numeric_limits<real>::max();
  real maxv = -std::numeric_limits<real>::max();

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp(curr_pos.y, 0, gridSize.y - 1); 
    const int32_t k0 = clamp(curr_pos.z, 0, 
                             (orig.is_3d() ? (gridSize.z - 1) : 1));
    const int32_t i1 = i0 + 1;
    const int32_t j1 = j0 + 1;
    const int32_t k1 = (orig.is_3d() ? (k0 + 1) : k0);
    if (!orig.isInBounds(Int3(i0, j0, k0), 0) ||
        !orig.isInBounds(Int3(i1, j1, k1), 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(i0, j0, k0, b));
    getMinMax(minv, maxv, orig(i1, j0, k0, b));
    getMinMax(minv, maxv, orig(i0, j1, k0, b));
    getMinMax(minv, maxv, orig(i1, j1, k0, b));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(i0, j0, k1, b));
      getMinMax(minv, maxv, orig(i1, j0, k1, b));
      getMinMax(minv, maxv, orig(i0, j1, k1, b)); 
      getMinMax(minv, maxv, orig(i1, j1, k1, b));
    }
  }

  dst = clamp(dst, minv, maxv);
  return dst;
}

inline real MacCormackClamp(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, real dval,
    const tfluids_(RealGrid)& orig, const tfluids_(RealGrid)& fwd, real dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  Int3 gridUpper = flags.getSize() - 1;

  dval = doClampComponent(gridUpper, dval, orig, fwd(i, j, k, b),
                          tfluids_(vec3)(i, j, k),
                          vel.getCentered(i, j, k, b) * dt, b);

  // Lookup forward/backward, round to closest NB.
  Int3 pos_fwd = toInt3(tfluids_(vec3)(i, j, k) +
                        tfluids_(vec3)(0.5, 0.5, 0.5) -
                        vel.getCentered(i, j, k, b) * dt);
  Int3 pos_bwd = toInt3(tfluids_(vec3)(i, j, k) +
                        tfluids_(vec3)(0.5, 0.5, 0.5) +
                        vel.getCentered(i, j, k, b) * dt);

  // Test if lookups point out of grid or into obstacle (note doClampComponent
  // already checks sides, below is needed for valid flags access).
  if (pos_fwd.x < 0 || pos_fwd.y < 0 || pos_fwd.z < 0 ||
      pos_bwd.x < 0 || pos_bwd.y < 0 || pos_bwd.z < 0 ||
      pos_fwd.x > gridUpper.x || pos_fwd.y > gridUpper.y ||
      ((pos_fwd.z > gridUpper.z) && flags.is_3d()) ||
      pos_bwd.x > gridUpper.x || pos_bwd.y > gridUpper.y ||
      ((pos_bwd.z > gridUpper.z) && flags.is_3d()) ||
      flags.isObstacle(pos_fwd, b) || flags.isObstacle(pos_bwd, b) ) {
    dval = fwd(i, j, k, b);
  }

  return dval;
}

static int tfluids_(Main_advectScalar)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* tensor_s =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_fwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* tensor_bwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));
  const std::string method = static_cast<std::string>(lua_tostring(L, 8));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 9));
  THTensor* tensor_s_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 10, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) src(tensor_s, is_3d);
  tfluids_(RealGrid) dst(tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  tfluids_(RealGrid) fwd(tensor_fwd, is_3d);
  tfluids_(RealGrid) bwd(tensor_bwd, is_3d); 


  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const bool is_levelset = false;  // We never advect them.
  const int order_space = 1;

  const int32_t nbatch = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  for (int32_t b = 0; b < nbatch; b++) {
    const int32_t bnd = 1;
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            if (order == 1) {
              dst(i, j, k, b) = 0;
            } else {
              fwd(i, j, k, b) = 0;
            }
            continue;
          }

          // Forward step.
          const real val = SemiLagrange(flags, vel, src, dt, is_levelset,
                                        order_space, i, j, k, b); 

          if (order == 1) {
            dst(i, j, k, b) = val;  // Store in the output array
          } else {
            fwd(i, j, k, b) = val;  // Store in the fwd array.
          }
        }
      }
    }

    if (order == 1) {
      // We're done. The forward Euler step is already in the output array.
    } else {

      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // beforemoving on).
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd(i, j, k, b) = 0;
              continue; 
            } 

            // Backwards step.
            bwd(i, j, k, b) = SemiLagrange(flags, vel, fwd, -dt, is_levelset,
                                           order_space, i, j, k, b);
          }
        }
      }

      // Now compute the correction.
      const real strength = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst(i, j, k, b) = MacCormackCorrect(flags, src, fwd, bwd, strength,
                                                is_levelset, i, j, k, b);
          }
        }
      }

      // Now perform clamping.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            const real dval = dst(i, j, k, b);
            dst(i, j, k, b) = MacCormackClamp(flags, vel, dval, src, fwd, dt,
                                              i, j, k, b);
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// advectVel
// *****************************************************************************

inline tfluids_(vec3) SemiLagrangeMAC(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, tfluids_(MACGrid)& src,
    real dt, int order_space, int32_t i, int32_t j, int32_t k, int32_t b) {
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const tfluids_(vec3) pos(static_cast<real>(i) + 0.5,
                           static_cast<real>(j) + 0.5,
                           static_cast<real>(k) + 0.5);

  tfluids_(vec3) xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const real vx = src.getInterpolatedComponentHi<0>(xpos, order_space, b);

  tfluids_(vec3) ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const real vy = src.getInterpolatedComponentHi<1>(ypos, order_space, b);

  real vz;
  if (vel.is_3d()) {
    tfluids_(vec3) zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi<2>(zpos, order_space, b);
  } else {
    vz = 0;
  }

  return tfluids_(vec3)(vx, vy, vz);
}

inline tfluids_(vec3) MacCormackCorrectMAC(
    tfluids_(FlagGrid)& flags, const tfluids_(MACGrid)& old,
    const tfluids_(MACGrid)& fwd, const tfluids_(MACGrid)& bwd,
    const real strength, int32_t i, int32_t j, int32_t k, int32_t b) {
  bool skip[3] = {false, false, false};

  if (!flags.isFluid(i, j, k, b)) {
    skip[0] = true;
    skip[1] = true;
    skip[2] = true;
  }

  // Note: in Manta code there's a isMAC boolean that is always true.
  if ((i > 0) && (!flags.isFluid(i - 1, j, k, b))) {
    skip[0] = true;
  }
  if ((j > 0) && (!flags.isFluid(i, j - 1, k, b))) {
    skip[1] = true;
  }
  if (flags.is_3d()) {
    if ((k > 0) && (!flags.isFluid(i, j, k - 1, b))) {
      skip[2] = true;
    }
  }

  tfluids_(vec3) dst(0, 0, 0);

  const int32_t dim = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < dim; ++c) {
    if (skip[c]) {
      dst(c) = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      dst(c) = fwd(i, j, k, c, b) + strength * 0.5 * (old(i, j, k, c, b) -
                                                      bwd(i, j, k, c, b));
    }
  }

  return dst;
}

template <int32_t c>
inline real doClampComponentMAC(
    const Int3& gridSize, real dst, const tfluids_(MACGrid)& orig,
    real fwd, const tfluids_(vec3)& pos, const tfluids_(vec3)& vel,
    int32_t b) {
  real minv = std::numeric_limits<real>::max();
  real maxv = -std::numeric_limits<real>::max();

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp(curr_pos.y, 0, gridSize.y - 1);
    const int32_t k0 = clamp(curr_pos.z, 0,
                             (orig.is_3d() ? (gridSize.z - 1) : 1));
    const int32_t i1 = i0 + 1;
    const int32_t j1 = j0 + 1;
    const int32_t k1 = (orig.is_3d() ? (k0 + 1) : k0);
    if (!orig.isInBounds(Int3(i0, j0, k0), 0) ||
        !orig.isInBounds(Int3(i1, j1, k1), 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(i0, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i0, j1, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j1, k0, c, b));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(i0, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i0, j1, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j1, k1, c, b));
    }
  }

  dst = clamp(dst, minv, maxv);
  return dst;
}

inline tfluids_(vec3) MacCormackClampMAC(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel,
    tfluids_(vec3) dval, const tfluids_(MACGrid)& orig,
    const tfluids_(MACGrid)& fwd, real dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  tfluids_(vec3) pos(static_cast<real>(i), static_cast<real>(j),
                     static_cast<real>(k));
  tfluids_(vec3) dfwd = fwd(i, j, k, b);
  Int3 gridUpper = flags.getSize() - 1;

  dval.x = doClampComponentMAC<0>(gridUpper, dval.x, orig, dfwd.x, pos,
                                  vel.getAtMACX(i, j, k, b) * dt, b);
  dval.y = doClampComponentMAC<1>(gridUpper, dval.y, orig, dfwd.y, pos,
                                  vel.getAtMACY(i, j, k, b) * dt, b);
  if (flags.is_3d()) {
    dval.z = doClampComponentMAC<2>(gridUpper, dval.z, orig, dfwd.z, pos,
                                    vel.getAtMACZ(i, j, k, b) * dt, b);
  } else {
    dval.z = 0;
  }

  // Note (from Manta): The MAC version currently does not check whether source 
  // points were inside an obstacle! (unlike centered version) this would need
  // to be done for each face separately to stay symmetric.
  
  return dval;
}

static int tfluids_(Main_advectVel)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_fwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_bwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 6));
  const std::string method = static_cast<std::string>(lua_tostring(L, 7));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 8));
  THTensor* tensor_u_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 9, torch_Tensor));

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const int order_space = 1;

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  tfluids_(MACGrid) orig(tensor_u, is_3d);
  tfluids_(MACGrid) dst(tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  tfluids_(MACGrid) fwd(tensor_fwd, is_3d);
  tfluids_(MACGrid) bwd(tensor_bwd, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            if (order == 1) {
              dst.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            } else {
              fwd.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            }
            continue;
          }

          // Forward step.
          tfluids_(vec3) val = SemiLagrangeMAC(flags, vel, orig, dt,
                                               order_space, i, j, k, b); 

          if (order == 1) {
            dst.setSafe(i, j, k, b, val);  // Store in the output array
          } else {
            fwd.setSafe(i, j, k, b, val);  // Store in the fwd array.
          }
        }
      }
    }

    if (order == 1) {
      // We're done. The forward Euler step is already in the output array.
    } else {

      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // before moving on).
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
              continue; 
            } 

            // Backwards step.
            bwd.setSafe(i, j, k, b, SemiLagrangeMAC(
                flags, vel, fwd, -dt, order_space, i, j, k, b));
          }
        }
      }

      // Now compute the correction.
      const real strength = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst.setSafe(i, j, k, b, MacCormackCorrectMAC(
                flags, orig, fwd, bwd, strength, i, j, k, b));
          }
        }
      }

      // Now perform clamping.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            const tfluids_(vec3) dval = dst(i, j, k, b);
            dst.setSafe(i, j, k, b, MacCormackClampMAC(
                flags, vel, dval, orig, fwd, dt, i, j, k, b));
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

static int tfluids_(Main_setWallBcsForward)(lua_State *L) {
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          const bool cur_fluid = flags.isFluid(i, j, k, b);
          const bool cur_obs = flags.isObstacle(i, j, k, b);
          if (!cur_fluid && !cur_obs) {
            continue;
          }
  
          // we use i > 0 instead of bnd=1 to check outer wall
          if (i > 0 && flags.isObstacle(i - 1, j, k, b)) {
            // TODO(tompson): Set to (potentially) non-zero obstacle velocity.
            vel(i, j, k, 0, b) = 0;
          }
          if (i > 0 && cur_obs && flags.isFluid(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) = 0;
          }
          if (j > 0 && flags.isObstacle(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) = 0;
          }
          if (j > 0 && cur_obs && flags.isFluid(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) = 0;
          }
  
          if (k > 0 && flags.isObstacle(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) = 0;
          }
          if (k > 0 && cur_obs && flags.isFluid(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) = 0;
          }
  
          if (cur_fluid) {
            if ((i > 0 && flags.isStick(i - 1, j, k, b)) ||
                (i < flags.xsize() - 1 && flags.isStick(i + 1, j, k, b))) {
              vel(i, j, k, 1, b) = 0;
              if (vel.is_3d()) {
                vel(i, j, k, 2, b) = 0;
              }
            }
            if ((j > 0 && flags.isStick(i, j - 1, k, b)) ||
                (j < flags.ysize() - 1 && flags.isStick(i, j + 1, k, b))) {
              vel(i, j, k, 0, b) = 0;
              if (vel.is_3d()) {
                vel(i, j, k, 2, b) = 0;
              }
            }
            if (vel.is_3d() &&
                ((k > 0 && flags.isStick(i, j, k - 1, b)) ||
                 (k < flags.zsize() - 1 && flags.isStick(i, j, k + 1, b)))) {
              vel(i, j, k, 0, b) = 0;
              vel(i, j, k, 1, b) = 0;
            }
          }
        }
      }
    } 
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

static int tfluids_(Main_velocityDivergenceForward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_u_div =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) rhs(tensor_u_div, is_3d);


  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reduction) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            rhs(i, j, k, b) = 0;
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            rhs(i, j, k, b) = 0;
            continue;
          }

          // compute divergence 
          // no flag checks: assumes vel at obstacle interfaces is set to zero.
          real div = 
              vel(i, j, k, 0, b) - vel(i + 1, j, k, 0, b) +
              vel(i, j, k, 1, b) - vel(i, j + 1, k, 1, b);
          if (is_3d) {
            div += (vel(i, j, k, 2, b) - vel(i, j, k + 1, 2, b));
          }
          rhs(i, j, k, b) = div;
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityDivergenceBackward
// *****************************************************************************

static int tfluids_(Main_velocityDivergenceBackward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
  THTensor* tensor_grad_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(RealGrid) grad_output(tensor_grad_output, is_3d);
  tfluids_(MACGrid) grad_u(tensor_grad_u, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();


  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    // Firstly, we're going to accumulate gradient contributions, so set
    // grad_u to 0.
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          grad_u.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
        }
      }
    }

    // Now accumulate gradients from across the output gradient.
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reductiion) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border in the forward pass, so they do
            // not contribute gradient.
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            // Blocked cells don't contribute gradient.
            continue;
          }

          // TODO(tompson): Can we restructure this into a gather rather than
          // a scatter? (it would mean multiple redundant lookups into flags,
          // but it might be faster...).
          const real go = grad_output(i, j, k, b);
#pragma omp atomic
          grad_u(i, j, k, 0, b) += go;
#pragma omp atomic
          grad_u(i + 1, j, k, 0, b) -= go;
#pragma omp atomic
          grad_u(i, j, k, 1, b) += go;
#pragma omp atomic
          grad_u(i, j + 1, k, 1, b) -= go;
          if (is_3d) {
#pragma omp atomic
            grad_u(i, j, k, 2, b) += go;
#pragma omp atomic
            grad_u(i, j, k + 1, 2, b) -= go;
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// emptyDomain
// *****************************************************************************

static int tfluids_(Main_emptyDomain)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 2));
  const int32_t bnd = static_cast<int32_t>(lua_tointeger(L, 3));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch  = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            flags(i, j, k, b) = TypeObstacle;
          } else {
            flags(i, j, k, b) = TypeFluid;
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// flagsToOccupancy
// *****************************************************************************

static int tfluids_(Main_flagsToOccupancy)(lua_State *L) {
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_occupancy=
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));

  const int32_t numel = THTensor_(numel)(tensor_flags);
  const real* pflags =
      reinterpret_cast<const real*>(THTensor_(data)(tensor_flags));
  real* pocc = reinterpret_cast<real*>(THTensor_(data)(tensor_occupancy));

  if (!THTensor_(isContiguous)(tensor_flags) ||
      !THTensor_(isContiguous)(tensor_occupancy)) {
    luaL_error(L, "ERROR: tensors are not contiguous!");
  }

  int32_t i;
  bool bad_cell = false;
#pragma omp parallel for private(i)
  for (i = 0; i < numel; i++) {
    const int32_t flag = static_cast<int32_t>(pflags[i]);
    if (flag == TypeFluid) {
      pocc[i] = 0;
    } else if (flag == TypeObstacle) {
      pocc[i] = 1;
    } else {
      bad_cell = true;  // There's no race cond because we'll only trigger once.
    }
  }

  if (bad_cell) {
    luaL_error(L, "ERROR: unsupported flag cell found!");
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

static int tfluids_(Main_velocityUpdateForward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
 
  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) pressure(tensor_p, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize(); 
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd || 
              j < bnd || j > ysize - 1 - bnd || 
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta doesn't touch the velocity on the boundaries (i.e.
            // it stays constant).
            continue;
          }

          if (flags.isFluid(i, j, k, b)) {   
            if (flags.isFluid(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) -= (pressure(i, j, k, b) -
                                     pressure(i - 1, j, k, b));
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j - 1, k, b));
            }
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j, k - 1, b));
            }
      
            if (flags.isEmpty(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) -= pressure(i, j, k, b);
            }
            if (flags.isEmpty(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) -= pressure(i, j, k, b);
            }
            if (is_3d && flags.isEmpty(i, j, k - 1, b)) {
              vel(i, j, k, 2, b) -= pressure(i, j, k, b);
            }
          }
          else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
            // don't change velocities in outflow cells   
            if (flags.isFluid(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) += pressure(i - 1, j, k, b);
            } else {
              vel(i, j, k, 0, b)  = 0.f;
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) += pressure(i, j - 1, k, b);
            } else {
              vel(i, j, k, 1, b)  = 0.f;
            }
            if (is_3d) {
              if (flags.isFluid(i, j, k - 1, b)) {
                vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
              } else {
                vel(i, j, k, 2, b)  = 0.f;
              }
            }
          }
        }
      }
    }
  }
  
  return 0;  // Recall: number of return values on the lua stack. 
}


// *****************************************************************************
// velocityUpdateBackward
// *****************************************************************************

static int tfluids_(Main_velocityUpdateBackward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  THTensor* tensor_grad_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(RealGrid) grad_p(tensor_grad_p, is_3d);
  tfluids_(MACGrid) grad_output(tensor_grad_output, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    // Firstly, we're going to accumulate gradient contributions, so set
    // grad_p to 0.
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          grad_p(i, j, k, b) = 0;
        }
      }
    }

    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta doesn't touch the velocity on the boundaries (i.e.
            // it stays constant and so has zero gradient).
            continue;
          }

          const tfluids_(vec3) go(grad_output(i, j, k, b));

          if (flags.isFluid(i, j, k, b)) {
            if (flags.isFluid(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) -= (p(i, j, k, b) - p(i - 1, j, k, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.x;
#pragma omp atomic
              grad_p(i - 1, j, k, b) += go.x;
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) -= (p(i, j, k, b) - p(i, j - 1, k, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.y;
#pragma omp atomic  
              grad_p(i, j - 1, k, b) += go.y;
            }
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              // fwd: vel(i, j, k, 2, b) -= (p(i, j, k, b) - p(i, j, k - 1, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.z;
#pragma omp atomic  
              grad_p(i, j, k - 1, b) += go.z;
            }

            if (flags.isEmpty(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.x;
            }
            if (flags.isEmpty(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.y;
            }
            if (is_3d && flags.isEmpty(i, j, k - 1, b)) {
              // fwd: vel(i, j, k, 2, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.z;
            }
          }
          else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
            // don't change velocities in outflow cells   
            if (flags.isFluid(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) += p(i - 1, j, k, b);
#pragma omp atomic
              grad_p(i - 1, j, k, b) += go.x;
            } else {
              // fwd: vel(i, j, k, 0, b)  = 0.f;
              // Output doesn't depend on p, so gradient is zero and so doesn't
              // contribute.
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) += p(i, j - 1, k, b);
#pragma omp atomic
              grad_p(i, j - 1, k, b) += go.y;
            } else {
              // fwd: vel(i, j, k, 1, b)  = 0.f;
              // Output doesn't depend on p, so gradient is zero and so doesn't
              // contribute.
            }
            if (is_3d) {
              if (flags.isFluid(i, j, k - 1, b)) {
                // fwd: vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
#pragma omp atomic
                grad_p(i, j, k - 1, b) += go.z;
              } else {
                // fwd: vel(i, j, k, 2, b)  = 0.f;
                // Output doesn't depend on p, so gradient is zero and so
                // doesn't contribute.
              }
            }
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// addBuoyancy
// *****************************************************************************
  
static int tfluids_(Main_addBuoyancy)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_density =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_gravity =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_strength =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const real dt = static_cast<real>(lua_tonumber(L, 6));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));

  if (tensor_gravity->nDimension != 1 || tensor_gravity->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }
  const real* gdata = THTensor_(data)(tensor_gravity);

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) factor(tensor_density, is_3d);

  // Note: We wont use the tensor_strength temp space for the C++ version.
  // It's just as fast (and easy) for us to wrap in a vec3.
  static_cast<void>(tensor_strength);
  const tfluids_(vec3) strength =
      tfluids_(vec3)(-gdata[0], -gdata[1], -gdata[2]) * (dt / flags.getDx());

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reductiion) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // No buoyancy on the border.
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            continue;
          }
          if (flags.isFluid(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) += (static_cast<real>(0.5) * strength.x *
                                (factor(i, j, k, b) + factor(i - 1, j, k, b)));
          }
          if (flags.isFluid(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) += (static_cast<real>(0.5) * strength.y *
                                (factor(i, j, k, b) + factor(i, j - 1, k, b)));
          }
          if (is_3d && flags.isFluid(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) += (static_cast<real>(0.5) * strength.z *
                                (factor(i, j, k, b) + factor(i, j, k - 1, b)));
          }
        }
      }
    }
  }
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// vorticityConfinement
// *****************************************************************************

inline void AddForceField(
    const tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel,
    const tfluids_(VecGrid)& force, int32_t i, int32_t j, int32_t k,
    int32_t b) {
  const bool curFluid = flags.isFluid(i, j, k, b);
  const bool curEmpty = flags.isEmpty(i, j, k, b);
  if (!curFluid && !curEmpty) {
    return;
  }

  if (flags.isFluid(i - 1, j, k, b) || 
      (curFluid && flags.isEmpty(i - 1, j, k, b))) {
    vel(i, j, k, 0, b) += (static_cast<real>(0.5) *
                        (force(i - 1, j, k, 0, b) + force(i, j, k, 0, b)));
  }

  if (flags.isFluid(i, j - 1, k, b) ||
      (curFluid && flags.isEmpty(i, j - 1, k, b))) {
    vel(i, j, k, 1, b) += (static_cast<real>(0.5) * 
                        (force(i, j - 1, k, 1, b) + force(i, j, k, 1, b)));
  }

  if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
      (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
    vel(i, j, k, 2, b) += (static_cast<real>(0.5) *
                        (force(i, j, k - 1, 2, b) + force(i, j, k, 2, b)));
  }
}

static int tfluids_(Main_vorticityConfinement)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  const real strength = static_cast<real>(lua_tonumber(L, 3));
  THTensor* tensor_centered =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* tensor_curl_norm =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));
  THTensor* tensor_force =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 7, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 8));


  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(VecGrid) centered(tensor_centered, is_3d);
  tfluids_(VecGrid) curl(tensor_curl, true);  // Alawys 3D.
  tfluids_(RealGrid) curl_norm(tensor_curl_norm, is_3d);
  tfluids_(VecGrid) force(tensor_force, is_3d);
  
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // First calculate the centered velocity.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            centered.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            continue;
          }
          centered.setSafe(i, j, k, b, vel.getCentered(i, j, k, b));
        }
      }
    }

    // Now calculate the curl and it's (l2) norm.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            curl.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            curl_norm(i, j, k, b) = 0;
            continue;
          }

          // Calculate the curl and it's (l2) norm.
          const tfluids_(vec3) cur_curl(centered.curl(i, j, k, b));
          curl.setSafe(i, j, k, b, cur_curl);
          curl_norm(i, j, k, b) = cur_curl.norm();
        }
      } 
    }

    // Now calculate the vorticity confinement force.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd || 
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Don't add force on the boundaries.
            force.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            continue;
          }

          tfluids_(vec3) grad(0, 0, 0);
          grad.x = static_cast<real>(0.5) * (curl_norm(i + 1, j, k, b) -
                                             curl_norm(i - 1, j, k, b));
          grad.y = static_cast<real>(0.5) * (curl_norm(i, j + 1, k, b) -
                                             curl_norm(i, j - 1, k, b));
          if (is_3d) {
            grad.z = static_cast<real>(0.5) * (curl_norm(i, j, k + 1, b) -
                                               curl_norm(i, j, k - 1, b));
          }
          grad.normalize();
          
          force.setSafe(i, j, k, b, tfluids_(vec3)::cross(
              grad, curl(i, j, k, b)) * strength);
        }   
      }
    }

    // Now apply the force.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            continue;
          }
          AddForceField(flags, vel, force, i, j, k, b);          
        }  
      }
    } 
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// drawVelocityField
// *****************************************************************************

static int tfluids_(Main_drawVelocityField)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const bool flip_y = static_cast<bool>(lua_toboolean(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  if (tensor_u->nDimension != 5) {
    luaL_error(L, "Input vector field should be 5D.");
  }

  tfluids_(MACGrid) vel(tensor_u, is_3d);

  const int32_t nbatch = vel.nbatch();
  const int32_t xsize = vel.xsize();
  const int32_t ysize = vel.ysize();
  const int32_t zsize = vel.zsize();

  if (nbatch > 1) {
    luaL_error(L, "input velocity field has more than one sample.");
  }

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glBegin(GL_LINES);
  const int32_t bnd = 1;
  for (int32_t b = 0; b < nbatch; b++) {
    for (int32_t z = 0; z < zsize; z++) {
      for (int32_t y = 0; y < ysize; y++) {
        for (int32_t x = 0; x < xsize; x++) {
          if (x < bnd || x > xsize - 1 - bnd ||
              y < bnd || y > ysize - 1 - bnd ||
              (is_3d && (z < bnd || z > zsize - 1 - bnd))) {
            continue;
          }
          tfluids_(vec3) v = vel.getCentered(x, y, z, b);
         
          // Velocity is in grids / second. But we need coordinates in [0, 1].
          v.x = v.x / static_cast<real>(xsize - 1);
          v.y = v.y / static_cast<real>(ysize - 1);
          v.z = is_3d ? v.z / static_cast<real>(zsize - 1) : 0;

          // Same for position.
          real px = static_cast<real>(x) / static_cast<real>(xsize - 1);
          real py = static_cast<real>(y) / static_cast<real>(ysize - 1);
          real pz =
              is_3d ? static_cast<real>(z) / static_cast<real>(zsize - 1) : 0;
          py = flip_y ? py : static_cast<real>(1) - py;
          v.y = flip_y ? -v.y : v.y;
          glColor4f(0.7f, 0.0f, 0.0f, 1.0f);
          glVertex3f(static_cast<float>(px),
                     static_cast<float>(py),
                     static_cast<float>(pz));
          glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
          glVertex3f(static_cast<float>(px + v.x),
                     static_cast<float>(py - v.y),
                     static_cast<float>(pz + v.z));
        }
      }
    }
  }
  glEnd();
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
}

// *****************************************************************************
// loadTensorTexture
// *****************************************************************************

static int tfluids_(Main_loadTensorTexture)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* im_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  if (im_tensor->nDimension != 2 && im_tensor->nDimension != 3) {
    luaL_error(L, "Input should be 2D or 3D.");
  }
  const int32_t tex_id = static_cast<int32_t>(luaL_checkinteger(L, 2));
  if (!lua_isboolean(L, 3)) {
    luaL_error(L, "3rd argument to loadTensorTexture should be boolean.");
  }
  const bool filter = lua_toboolean(L, 3);
  if (!lua_isboolean(L, 4)) {
    luaL_error(L, "4rd argument to loadTensorTexture should be boolean.");
  }
  const bool flip_y = lua_toboolean(L, 4);

  const bool grey = im_tensor->nDimension == 2;
  const int32_t nchan = grey ? 1 : im_tensor->size[0];
  const int32_t h = grey ? im_tensor->size[0] : im_tensor->size[1];
  const int32_t w = grey ? im_tensor->size[1] : im_tensor->size[2];

  if (nchan != 1 && nchan != 3) {
    luaL_error(L, "Only 3 or 1 channels is supported.");
  }

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex_id);

  if (filter) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  } else {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  const real* im_tensor_data = THTensor_(data)(im_tensor);

  // We need to either: a) swizzle the RGB data, b) convert from double to float
  // or c) convert to RGB greyscale for single channel textures). For c) we
  // could use a swizzle mask, but this complicates alpha blending, so it's
  // easier to just convert always at the cost of a copy (which is parallel and
  // fast).

  std::unique_ptr<float[]> fdata(new float[h * w * 4]);
  int32_t c, u, v;
#pragma omp parallel for private(v, u, c) collapse(3)
  for (v = 0; v < h; v++) {
    for (u = 0; u < w; u++) {
      for (c = 0; c < 4; c++) {
        if (c == 3) {
          // OpenMP requires perfectly nested loops, so we need to include the
          // alpha chan set like this.
          fdata[v * 4 * w + u * 4 + c] = 1.0f;
        } else {
          const int32_t csrc = (c < nchan) ? c : 0;
          const int32_t vsrc = flip_y ? (h - v - 1) : v;
          fdata[v * 4 * w + u * 4 + c] =
              static_cast<float>(im_tensor_data[csrc * w * h + vsrc * w + u]);
        }
      }
    }
  }

  const GLint level = 0;
  const GLint internalformat = GL_RGBA32F;
  const GLint border = 0;
  const GLenum format = GL_RGBA;
  const GLenum type = GL_FLOAT;
  glTexImage2D(GL_TEXTURE_2D, level, internalformat, w, h, border,
               format, type, fdata.get());
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
}

// *****************************************************************************
// volumetricUpsamplingNearestForward
// *****************************************************************************

static int tfluids_(Main_volumetricUpSamplingNearestForward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));

  if (input->nDimension != 5 || output->nDimension != 5) {
    luaL_error(L, "ERROR: input and output must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zsize = input->size[2];
  const int32_t ysize = input->size[3];
  const int32_t xsize = input->size[4];

  if (output->size[0] != nbatch || output->size[1] != nfeat ||
      output->size[2] != zsize * ratio || output->size[3] != ysize * ratio ||
      output->size[4] != xsize * ratio) {
    luaL_error(L, "ERROR: input : output size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  real* output_data = THTensor_(data)(output);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zsize * ratio; z++) {
        for (y = 0; y < ysize * ratio; y++) {
          for (x = 0; x < xsize * ratio; x++) {
            const int64_t iout = output->stride[0] * b + output->stride[1] * f +
                output->stride[2] * z +
                output->stride[3] * y + 
                output->stride[4] * x;
            const int64_t iin = input->stride[0] * b + input->stride[1] * f +
                input->stride[2] * (z / ratio) +
                input->stride[3] * (y / ratio) +
                input->stride[4] * (x / ratio);
            output_data[iout] = input_data[iin];
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// volumetricUpsamplingNearestBackward
// *****************************************************************************

static int tfluids_(Main_volumetricUpSamplingNearestBackward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  if (input->nDimension != 5 || grad_output->nDimension != 5 ||
      grad_input->nDimension != 5) {
    luaL_error(L, "ERROR: input, gradOutput and gradInput must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zsize = input->size[2];
  const int32_t ysize = input->size[3];
  const int32_t xsize = input->size[4];

  if (grad_output->size[0] != nbatch || grad_output->size[1] != nfeat ||
      grad_output->size[2] != zsize * ratio ||
      grad_output->size[3] != ysize * ratio ||
      grad_output->size[4] != xsize * ratio) {
    luaL_error(L, "ERROR: input : gradOutput size mismatch.");
  }

  if (grad_input->size[0] != nbatch || grad_input->size[1] != nfeat ||
      grad_input->size[2] != zsize || grad_input->size[3] != ysize ||
      grad_input->size[4] != xsize) {
    luaL_error(L, "ERROR: input : gradInput size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  const real* grad_output_data = THTensor_(data)(grad_output);
  real * grad_input_data = THTensor_(data)(grad_input);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zsize; z++) {
        for (y = 0; y < ysize; y++) {
          for (x = 0; x < xsize; x++) {
            const int64_t iout = grad_input->stride[0] * b +
                grad_input->stride[1] * f +
                grad_input->stride[2] * z +
                grad_input->stride[3] * y +
                grad_input->stride[4] * x;
            float sum = 0;
            // Now accumulate gradients from the upsampling window.
            for (int32_t zup = 0; zup < ratio; zup++) {
              for (int32_t yup = 0; yup < ratio; yup++) {
                for (int32_t xup = 0; xup < ratio; xup++) {
                  const int64_t iin = grad_output->stride[0] * b +
                      grad_output->stride[1] * f +
                      grad_output->stride[2] * (z * ratio + zup) +
                      grad_output->stride[3] * (y * ratio + yup) +
                      grad_output->stride[4] * (x * ratio + xup);
                  sum += grad_output_data[iin];
                }
              }
            }
            grad_input_data[iout] = sum;
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// rectangularBlur
// *****************************************************************************

// This is a convolution with a rectangular kernel where we treat the kernel as
// an impulse train. It decouples the blur kernel size to the runtime
// resulting in an O(npix) transform (very fast even for massive kernels sizes).
static inline void DoRectangularBlurAlongAxis(
    const real* src, const int32_t size, const int32_t stride,
    const int32_t rad, real* dst) {
  // Initialize with the sum of the first pixel rad + 1 times.
  real val = src[0] * static_cast<real>(rad + 1);
  // Now accumulate the first rad - 1 elements. These two contributions
  // effectively start with the center pixel at i = -1, where we clamp the
  // edge values.
  for (int32_t i = 0; i < size && i < rad; i++) {
    val += src[i * stride];
  }

  // Now beging the algorithm.
  const real mul_const = static_cast<real>(1) / static_cast<real>(rad * 2 + 1);
  for (int32_t i = 0; i < size; i++) {
    // Move the current position over one by:
    // 1. Subtracting off the pixel 1 radius - 1 back.
    const int32_t iminus = std::max(0, i - rad - 1);
    val -= src[iminus * stride];
    // 2. Adding the pixel 1 radius forward.
    const int32_t iplus = std::min(size - 1, i + rad);
    val += src[iplus * stride];

    // Now divide by the number of output elements and set the output value.
    dst[i * stride] = val * mul_const;
  }
}

static int tfluids_(Main_rectangularBlur)(lua_State *L) {
  THTensor* src_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const int32_t blur_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THTensor* dst_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tmp_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));

  if (src_tensor->nDimension != 5 || dst_tensor->nDimension != 5 ||
      tmp_tensor->nDimension != 5) {
    luaL_error(L, "ERROR: src and dst must be dim 5");
  }

  const int32_t bsize = src_tensor->size[0];
  const int32_t fsize = src_tensor->size[1];
  const int32_t zsize = src_tensor->size[2];
  const int32_t ysize = src_tensor->size[3];
  const int32_t xsize = src_tensor->size[4];

  const int32_t bstride = src_tensor->stride[0];
  const int32_t fstride = src_tensor->stride[1];
  const int32_t zstride = src_tensor->stride[2];
  const int32_t ystride = src_tensor->stride[3];
  const int32_t xstride = src_tensor->stride[4];

  const real* src = THTensor_(data)(src_tensor);
  real* dst = THTensor_(data)(dst_tensor);
  real* tmp = THTensor_(data)(tmp_tensor);

  const real* cur_src = src;
  real* cur_dst = is_3d ? dst : tmp;

  int32_t b, f, z, y, x;
  if (is_3d) {
    // Do the blur in the z-dimension.
#pragma omp parallel for private(b, f, y, x) collapse(4)
    for (b = 0; b < bsize; b++) {
      for (f = 0; f < fsize; f++) {
        for (y = 0; y < ysize; y++) {
          for (x = 0; x < xsize; x++) {
            const real* in = &cur_src[b * bstride + f * fstride + y * ystride +
                                      x * xstride];
            real* out = &cur_dst[b * bstride + f * fstride + y * ystride +
                                 x * xstride];
            DoRectangularBlurAlongAxis(in, zsize, zstride, blur_rad, out);
          }
        }
      }
    }

    cur_src = dst;
    cur_dst = tmp;
  }
  // Do the blur in the y-dimension
#pragma omp parallel for private(b, f, z, x) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (f = 0; f < fsize; f++) {
      for (z = 0; z < zsize; z++) {
        for (x = 0; x < xsize; x++) {
          const real* in = &cur_src[b * bstride + f * fstride + z * zstride +
                                    x * xstride];
          real* out = &cur_dst[b * bstride + f * fstride + z * zstride +
                               x * xstride];
          DoRectangularBlurAlongAxis(in, ysize, ystride, blur_rad, out);
        }
      }
    }
  }

  cur_src = tmp;
  cur_dst = dst;

  // Do the blur in the x-dimension
#pragma omp parallel for private(b, f, z, y) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (f = 0; f < fsize; f++) { 
      for (z = 0; z < zsize; z++) {
        for (y = 0; y < ysize; y++) { 
          const real* in = &cur_src[b * bstride + f * fstride + z * zstride +
                                    y * ystride];
          real* out = &cur_dst[b * bstride + f * fstride + z * zstride +
                               y * ystride];
          DoRectangularBlurAlongAxis(in, xsize, xstride, blur_rad, out);
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// signedDistanceField
// *****************************************************************************

static int tfluids_(Main_signedDistanceField)(lua_State *L) {
  THTensor* flag_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const int32_t search_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THTensor* dst_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  tfluids_(FlagGrid) flags(flag_tensor, is_3d);
  tfluids_(RealGrid) dst(dst_tensor, is_3d);

  const int32_t bsize = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (z = 0; z < zsize; z++) {
      for (y = 0; y < ysize; y++) {
        for (x = 0; x < xsize; x++) {
          if (flags.isObstacle(x, y, z, b)) {
            dst(x, y, z, b) = 0;
            continue;
          }
          real dist_sq = static_cast<real>(search_rad * search_rad);
          const int32_t zmin = std::max(0, z - search_rad);;
          const int32_t zmax = std::min(zsize - 1, z + search_rad);
          const int32_t ymin = std::max(0, y - search_rad);;
          const int32_t ymax = std::min(ysize - 1, y + search_rad);
          const int32_t xmin = std::max(0, x - search_rad);;
          const int32_t xmax = std::min(xsize - 1, x + search_rad);
          for (int32_t zsearch = zmin; zsearch <= zmax; zsearch++) {
            for (int32_t ysearch = ymin; ysearch <= ymax; ysearch++) {
              for (int32_t xsearch = xmin; xsearch <= xmax; xsearch++) {
                if (flags.isObstacle(xsearch, ysearch, zsearch, b)) {
                  const real cur_dist_sq = ((z - zsearch) * (z - zsearch) +
                                            (y - ysearch) * (y - ysearch) +
                                            (x - xsearch) * (x - xsearch));
                  if (dist_sq > cur_dist_sq) {
                    dist_sq = cur_dist_sq;
                  }
                }
              }
            }
          }
          dst(x, y, z, b) = std::sqrt(dist_sq);
        }
      }
    }
  }


  return 0;
}

// *****************************************************************************
// solveLinearSystemPCG
// *****************************************************************************

static int tfluids_(Main_solveLinearSystemPCG)(lua_State *L) {
  luaL_error(L, "ERROR: solveLinearSystemPCG not defined for CPU tensors.");
  return 0;
}

// *****************************************************************************
// Init methods
// *****************************************************************************

static const struct luaL_Reg tfluids_(Main__) [] = {
  {"advectScalar", tfluids_(Main_advectScalar)},
  {"advectVel", tfluids_(Main_advectVel)},
  {"setWallBcsForward", tfluids_(Main_setWallBcsForward)},
  {"vorticityConfinement", tfluids_(Main_vorticityConfinement)},
  {"addBuoyancy", tfluids_(Main_addBuoyancy)},
  {"drawVelocityField", tfluids_(Main_drawVelocityField)},
  {"loadTensorTexture", tfluids_(Main_loadTensorTexture)},
  {"velocityUpdateForward", tfluids_(Main_velocityUpdateForward)},
  {"velocityUpdateBackward", tfluids_(Main_velocityUpdateBackward)},
  {"velocityDivergenceForward", tfluids_(Main_velocityDivergenceForward)},
  {"velocityDivergenceBackward", tfluids_(Main_velocityDivergenceBackward)},
  {"emptyDomain", tfluids_(Main_emptyDomain)},
  {"flagsToOccupancy", tfluids_(Main_flagsToOccupancy)},
  {"solveLinearSystemPCG", tfluids_(Main_solveLinearSystemPCG)},
  {"volumetricUpSamplingNearestForward",
   tfluids_(Main_volumetricUpSamplingNearestForward)},
  {"volumetricUpSamplingNearestBackward",
   tfluids_(Main_volumetricUpSamplingNearestBackward)},
  {"rectangularBlur", tfluids_(Main_rectangularBlur)},
  {"signedDistanceField", tfluids_(Main_signedDistanceField)},
  {NULL, NULL}  // NOLINT
};

void tfluids_(Main_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, tfluids_(Main__), "tfluids");
}

#endif  // TH_GENERIC_FILE
