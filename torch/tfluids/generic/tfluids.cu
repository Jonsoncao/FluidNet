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

#include <TH.h>
#include <THC.h>
#include <luaT.h>

#include <assert.h>
#include <cusparse.h>
#include <float.h>
#include <algorithm>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

#include "generic/cell_type.h"
#include "generic/grid.cu.h"
#include "generic/int3.cu.h"
#include "generic/vec3.cu.h"

const int threads_per_block = 512;  // Might need 256 for old SM.
const int64_t cuda_num_threads = 1024;  // Might need 256 for old SM.

// This is REALLY ugly. But unfortunately cutorch_getstate() in
// cutorch/torch/util.h is not exposed externally. We could call
// cutorch.getState() from lua and pass in the struct into all the tfluids c
// functions (as Soumith did with nn and cunn), but I think this is also just
// as ugly. Instead lets just redefine cutorch_getstate and hope nothing
// breaks :-(

struct THCState* cutorch_getstate(lua_State* L) {
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "_state");
  struct THCState* state = reinterpret_cast<THCState*>(lua_touserdata(L, -1));
  lua_pop(L, 2);
  return state;
}

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      THError("CUSPARSE Library initialization failed");
    }
  }
}

// *****************************************************************************
// LaunchKernel
// *****************************************************************************

// A simple helper function to reduce the amount of boiler plate code required
// to launch a kernel (it also cuts down the number of potential bugs).
// 
// All our kernels use an unknown number of parameters, so we'll need to
// pass in a function pointer with the correct signature as well as the
// arg lists.
//
// @template TFuncPtr: kernel func ptr. The compiler will autocomplete this!
// @template Args: Again, you do not need to define it (see emptyDomain).
// @param: func - the kernel function to call.
// @param: domain - the Tensor that defines the domain of the kernel (usually
// the output tensor).
// @param: args - the variable size argument list that the kernel takes as
// input.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernel(lua_State *L, TFuncPtr func,
                         const THCudaTensor* domain, Args... args) {
  THCState* state = cutorch_getstate(L);
  if (domain->nDimension != 5) {
    luaL_error(L, "input tensor for kernel domain is not 5D.");
  }
  const int xsize = domain->size[4];
  const int ysize = domain->size[3];
  const int zsize = domain->size[2];
  const int csize = domain->size[1];
  const int bsize = domain->size[0];

  // Create the kernel grid and block sizes.
  // TODO(tompson): What if csize is 1 (i.e. scalar domains). Is this slower?
  int nplane = xsize * ysize * zsize;
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), csize, bsize);
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  // Call the function.
  func<<<grid_size, block_size, 0, THCState_getCurrentStream(state)>>>(args...);
}

// Same as above, but on a one of our Grid objects.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernel(lua_State *L, TFuncPtr func,
                         const CudaGridBase& domain, Args... args) {
  THCState* state = cutorch_getstate(L);
  const int xsize = domain.xsize();
  const int ysize = domain.ysize();
  const int zsize = domain.zsize();
  const int csize = domain.nchan();
  const int bsize = domain.nbatch();

  // Create the kernel grid and block sizes.
  // TODO(tompson): What if csize is 1 (i.e. scalar domains). Is this slower?
  int nplane = xsize * ysize * zsize;
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), csize, bsize);
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  // Call the function.
  func<<<grid_size, block_size, 0, THCState_getCurrentStream(state)>>>(args...);
  THCudaCheck(cudaGetLastError());
}

inline int64_t GetBlocks(const int64_t n) {
  return (n + cuda_num_threads - 1) / cuda_num_threads;
}

// This method will launch a kernel over the entire domain numel.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernelLoop(lua_State *L, TFuncPtr func,
                             const CudaGridBase& domain, Args... args) {
  THCState* state = cutorch_getstate(L);

  // Call the function.
  // const int64_t numel = THCudaTensor_nElement(state, domain);
  const int64_t numel = domain.numel();
  func<<<GetBlocks(numel), cuda_num_threads, 0,
         THCState_getCurrentStream(state)>>>(args...);
  THCudaCheck(cudaGetLastError());
}

// Assumes you're iterating over a scalar domain (i.e nchan = 1 for the domain
// you're iterating over). The LaunchKernelLoop forces this since you cannot
// specify a nchan.
__device__ __forceinline__ void PntIdToScalarIndices(
    const int32_t nbatch, const int32_t zsize, const int32_t ysize,
    const int32_t xsize, const int32_t& pnt_id, int32_t& batch,
    int32_t& k, int32_t& j, int32_t& i) {
  i = pnt_id % xsize;
  j = (pnt_id / xsize) % ysize;
  k = (pnt_id / xsize / ysize) % zsize;
  batch = (pnt_id / xsize / ysize / zsize);
}

// CUDA: grid stride looping.
// This strategy comes from similar code in the cunn library.
#define CUDA_KERNEL_LOOP(numel, pnt_id) \
  for (int32_t pnt_id = blockIdx.x * blockDim.x + threadIdx.x; \
       pnt_id < (numel); \
       pnt_id += blockDim.x * gridDim.x)

// *****************************************************************************
// GetKernelIndices
// *****************************************************************************

// Another helper function to get back the batch, chan, k, j, i indices in a
// kernel launch by the LaunchKernel function above.
//
// If GetKernelIndices returns true, then the current kernel is out of the
// domain (and so you should just exist the kernel). This happens because
// the tensor may not fill up the last grid.
//
// Note, you should ALWAYS pass in the same size tensor as the tensor you used
// to call the kernel in LaunchKernel's domain parameter.
__device__ __forceinline__ bool GetKernelIndices(
    const THCDeviceTensor<float, 5>& domain, int32_t& batch, int32_t& chan,
    int32_t& k, int32_t& j, int32_t& i) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  chan = blockIdx.y;  // Should always be zero.
  batch = blockIdx.z;
  if (pnt_id >= (domain.getSize(2) * domain.getSize(3) * domain.getSize(4))) {
    return true;
  }
  i = pnt_id % domain.getSize(4);
  j = (pnt_id / domain.getSize(4)) % domain.getSize(3);
  k = pnt_id / (domain.getSize(3) * domain.getSize(4));
  return false;
}

// Same as above but on one of our Grid objects.
__device__ __forceinline__ bool GetKernelIndices(
    const CudaGridBase& domain, int32_t& batch, int32_t& chan, int32_t& k,
    int32_t& j, int32_t& i) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  chan = blockIdx.y;  // Should always be zero.
  batch = blockIdx.z;
  if (pnt_id >= (domain.zsize() * domain.ysize() * domain.xsize())) {
    return true;
  }
  i = pnt_id % domain.xsize();
  j = (pnt_id / domain.xsize()) % domain.ysize();
  k = pnt_id / (domain.ysize() * domain.xsize());
  return false;
}

// *****************************************************************************
// advectScalar
// *****************************************************************************

__global__ void SemiLagrange(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src, 
    CudaRealGrid dst, const float dt, const bool is_levelset,
    const int32_t order_space, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst(i, j, k, b) = 0;
    return;
  }

  CudaVec3 pos = (CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f) -
                  vel.getCentered(i, j, k, b) * dt);
  dst(i, j, k, b) = src.getInterpolatedHi(pos, order_space, b);
}

__global__ void SemiLagrangeLoop(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const bool is_levelset,
    const int32_t order_space, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst(i, j, k, b) = 0;
    return;
  }

  CudaVec3 pos = (CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f) -
                  vel.getCentered(i, j, k, b) * dt);
  dst(i, j, k, b) = src.getInterpolatedHi(pos, order_space, b);
}

__global__ void MacCormackCorrect(
    CudaFlagGrid flags, CudaRealGrid old, CudaRealGrid fwd,
    CudaRealGrid bwd, CudaRealGrid dst, const float strength,
    const bool is_levelset) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  float val = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    val += strength * 0.5f * (old(i, j, k, b) - bwd(i, j, k, b));
  }

  dst(i, j, k, b) = val;
}

__device__ __forceinline__ void getMinMax(float& minv, float& maxv,
                                          const float& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
}

template <typename T>
__device__ __forceinline__ T clamp(const T val, const T vmin, const T vmax) {
  if (val < vmin) {
    return vmin;
  }
  if (val > vmax) {
    return vmax;
  }
  return val; 
}

__device__ __forceinline__ float doClampComponent(
    const Int3& gridSize, float dst, CudaRealGrid orig, const float fwd,
    CudaVec3 pos, CudaVec3 vel, const int32_t b) {
  float minv = CUDART_INF_F;
  float maxv = -CUDART_INF_F;

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp<int32_t>(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp<int32_t>(curr_pos.y, 0, gridSize.y - 1); 
    const int32_t k0 = clamp<int32_t>(curr_pos.z, 0, 
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

  return clamp<float>(dst, minv, maxv);
}

__global__ void MacCormackClamp(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid dst,
    CudaRealGrid orig, CudaRealGrid fwd, const float dt, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  Int3 gridUpper = flags.getSize() - 1;
  float dval = dst(i, j, k, b);

  dval = doClampComponent(gridUpper, dval, orig, fwd(i, j, k, b),
                          CudaVec3(i, j, k),
                          vel.getCentered(i, j, k, b) * dt, b);

  // Lookup forward/backward, round to closest NB.
  Int3 pos_fwd = toInt3(CudaVec3(i, j, k) +
                        CudaVec3(0.5f, 0.5f, 0.5f) -
                        vel.getCentered(i, j, k, b) * dt);
  Int3 pos_bwd = toInt3(CudaVec3(i, j, k) +
                        CudaVec3(0.5f, 0.5f, 0.5f) +
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

  dst(i, j, k, b) = dval;
}

static int tfluids_CudaMain_advectScalar(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  float dt = static_cast<float>(lua_tonumber(L, 1));
  THCudaTensor* tensor_s = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_fwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* tensor_bwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));
  const std::string method = static_cast<std::string>(lua_tostring(L, 8));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 9));
  THCudaTensor* tensor_s_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 10, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid src = toCudaRealGrid(state, tensor_s, is_3d);
  CudaRealGrid dst = toCudaRealGrid(state, tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaRealGrid fwd = toCudaRealGrid(state, tensor_fwd, is_3d);
  CudaRealGrid bwd = toCudaRealGrid(state, tensor_bwd, is_3d);

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const bool is_levelset = false;  // We never advect them.
  const int32_t order_space = 1;

  // Do the forward step.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (order == 1) {
    LaunchKernel(L, &SemiLagrange, flags,
                 flags, vel, src, dst, dt, is_levelset, order_space, bnd);
    // We're done. The forward Euler step is already in the output array.
    return 0;
  } else {
    LaunchKernel(L, &SemiLagrange, flags,
                 flags, vel, src, fwd, dt, is_levelset, order_space, bnd);
  }

  // Do the backwards step.
  LaunchKernel(L, &SemiLagrange, flags,
               flags, vel, fwd, bwd, -dt, is_levelset, order_space, bnd);

  // Perform the correction.
  const float strength = 1.0f;
  LaunchKernel(L, &MacCormackCorrect, flags,
               flags, src, fwd, bwd, dst, strength, is_levelset);

  // Perform clamping.
  LaunchKernel(L, &MacCormackClamp, flags,
               flags, vel, dst, src, fwd, dt, bnd);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// advectVel
// *****************************************************************************
__global__ void SemiLagrangeMAC(
    CudaFlagGrid flags, CudaMACGrid vel, CudaMACGrid src,
    CudaMACGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }

  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const CudaVec3 pos(static_cast<float>(i) + 0.5f,
                     static_cast<float>(j) + 0.5f,
                     static_cast<float>(k) + 0.5f);

  CudaVec3 xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const float vx = src.getInterpolatedComponentHi<0>(xpos, order_space, b);

  CudaVec3 ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const float vy = src.getInterpolatedComponentHi<1>(ypos, order_space, b);

  float vz;
  if (vel.is_3d()) {
    CudaVec3 zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi<2>(zpos, order_space, b);
  } else {
    vz = 0;
  }

  dst.setSafe(i, j, k, b, CudaVec3(vx, vy, vz));
}

__global__ void MacCormackCorrectMAC(
    CudaFlagGrid flags, CudaMACGrid old, CudaMACGrid fwd, 
    CudaMACGrid bwd, CudaMACGrid dst, const float strength) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

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

  CudaVec3 val(0, 0, 0);
 
  const int32_t nchan = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < nchan; ++c) {
    if (skip[c]) {
      val(c) = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      val(c) = fwd(i, j, k, c, b) + strength * 0.5f * (old(i, j, k, c, b) -
                                                       bwd(i, j, k, c, b));
    }
  }

  dst.setSafe(i, j, k, b, val);
}

template <int32_t c>
__device__ __forceinline__ float doClampComponentMAC(
    const Int3& gridSize, float dst, const CudaMACGrid& orig,
    float fwd, const CudaVec3& pos, const CudaVec3& vel,
    int32_t b) {
  float minv = CUDART_INF_F;
  float maxv = -CUDART_INF_F;

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp<int32_t>(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp<int32_t>(curr_pos.y, 0, gridSize.y - 1);
    const int32_t k0 = clamp<int32_t>(curr_pos.z, 0,
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

  return clamp<float>(dst, minv, maxv);
}

__global__ void MacCormackClampMAC(
    CudaFlagGrid flags, CudaMACGrid vel, CudaMACGrid dst,
    CudaMACGrid orig, CudaMACGrid fwd, const float dt, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  CudaVec3 pos(static_cast<float>(i), static_cast<float>(j),
               static_cast<float>(k));
  CudaVec3 dval = dst(i, j, k, b);
  CudaVec3 dfwd = fwd(i, j, k, b);
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
  
  dst.setSafe(i, j, k, b, dval);
}

static int tfluids_CudaMain_advectVel(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const float dt = static_cast<float>(lua_tonumber(L, 1));
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_fwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_bwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 6));
  const std::string method = static_cast<std::string>(lua_tostring(L, 7));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 8));
  THCudaTensor* tensor_u_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 9, "torch.CudaTensor"));

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }

  const int32_t order = method == "euler" ? 1 : 2;
  const int32_t order_space = 1;

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  CudaMACGrid src = toCudaMACGrid(state, tensor_u, is_3d);
  CudaMACGrid dst = toCudaMACGrid(state, tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaMACGrid fwd = toCudaMACGrid(state, tensor_fwd, is_3d);
  CudaMACGrid bwd = toCudaMACGrid(state, tensor_bwd, is_3d);

  // Do the forward step.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (order == 1) {
    LaunchKernel(L, &SemiLagrangeMAC, flags,
                 flags, vel, src, dst, dt, order_space, bnd);
    // We're done. The forward Euler step is already in the output array.
    return 0;
  } else {
    LaunchKernel(L, &SemiLagrangeMAC, flags,
                 flags, vel, src, fwd, dt, order_space, bnd);
  }

  // Do the backwards step.
  LaunchKernel(L, &SemiLagrangeMAC, flags,
               flags, vel, fwd, bwd, -dt, order_space, bnd);

  // Perform the correction.
  const float strength = 1.0f;
  LaunchKernel(L, &MacCormackCorrectMAC, flags,
               flags, src, fwd, bwd, dst, strength);

  // Perform clamping.
  LaunchKernel(L, &MacCormackClampMAC, flags,
               flags, vel, dst, src, fwd, dt, bnd);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

__global__ void setWallBcsForward(CudaFlagGrid flags, CudaMACGrid vel) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  const bool cur_fluid = flags.isFluid(i, j, k, b);
  const bool cur_obs = flags.isObstacle(i, j, k, b);
  if (!cur_fluid && !cur_obs) {
    return;
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

static int tfluids_CudaMain_setWallBcsForward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &setWallBcsForward, flags,
               flags, vel);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

__global__ void velocityDivergenceForward(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid rhs, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
 
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    rhs(i, j, k, b) = 0;
    return;
  }

  if (!flags.isFluid(i, j, k, b)) {
    rhs(i, j, k, b) = 0;
    return;
  }

  // compute divergence 
  // no flag checks: assumes vel at obstacle interfaces is set to zero.
  float div = vel(i, j, k, 0, b) - vel(i + 1, j, k, 0, b) +
              vel(i, j, k, 1, b) - vel(i, j + 1, k, 1, b);
  if (flags.is_3d()) {
    div += (vel(i, j, k, 2, b) - vel(i, j, k + 1, 2, b));
  }
  rhs(i, j, k, b) = div;
}

static int tfluids_CudaMain_velocityDivergenceForward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_u_div = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid rhs = toCudaRealGrid(state, tensor_u_div, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &velocityDivergenceForward, flags,
               flags, vel, rhs, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityDivergenceBackward
// *****************************************************************************

__global__ void velocityDivergenceBackward(
    CudaFlagGrid flags, CudaMACGrid grad_u, CudaRealGrid grad_output,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border in the forward pass, so they do
    // not contribute gradient.
    return;
  }

  if (!flags.isFluid(i, j, k, b)) {
    // Blocked cells don't contribute gradient.
    return;
  }

  // TODO(tompson): I'm sure these atomic add calls are slow! We should
  // probably change this from a scatter to a gather op to avoid having to use
  // them at all.
  // (NVIDIA state that atomic operations on global memory are extremely slow)
  // but on shared memory it is OK. So we could copy to shared first, use
  // atomic ops there then use a small number of atomic ops back to global mem
  // (probably rewriting it as a gather would be easier).
  const float go = grad_output(i, j, k, b);
  atomicAdd(&grad_u(i, j, k, 0, b), go);
  atomicAdd(&grad_u(i + 1, j, k, 0, b), -go);
  atomicAdd(&grad_u(i, j, k, 1, b), go);
  atomicAdd(&grad_u(i, j + 1, k, 1, b), -go); 
  if (flags.is_3d()) {
    atomicAdd(&grad_u(i, j, k, 2, b), go);
    atomicAdd(&grad_u(i, j, k + 1, 2, b), -go); 
  } 
}

static int tfluids_CudaMain_velocityDivergenceBackward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
  THCudaTensor* tensor_grad_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid grad_u = toCudaMACGrid(state, tensor_grad_u, is_3d);
  CudaRealGrid grad_output = toCudaRealGrid(state, tensor_grad_output, is_3d);

  // Firstly, we're going to accumulate gradient contributions, so set
  // grad_u to 0.
  THCudaTensor_zero(state, tensor_grad_u);

  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &velocityDivergenceBackward, flags,
               flags, grad_u, grad_output, bnd);
   
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// emptyDomain
// *****************************************************************************

__global__ void emptyDomainLoop(
    CudaFlagGrid flags, const bool is_3d, const int32_t bnd,
    const int32_t nbatch, const int32_t zsize, const int32_t ysize,
    const int32_t xsize, const int32_t numel) {
  int32_t b, k, j, i;
  CUDA_KERNEL_LOOP(numel, pnt_id) {
    PntIdToScalarIndices(nbatch, zsize, ysize, xsize, pnt_id, b, k, j, i);  
    if (i < bnd || i > flags.xsize() - 1 - bnd ||
        j < bnd || j > flags.ysize() - 1 - bnd ||
        (is_3d && (k < bnd || k > flags.zsize() - 1 - bnd))) {
      flags(i, j, k, b) = TypeObstacle;
    } else {
      flags(i, j, k, b) = TypeFluid;
    }
  }
}

__global__ void emptyDomain(
     CudaFlagGrid flags, const bool is_3d, const int32_t bnd) {
  int32_t b, dim, k, j, i;
  if (GetKernelIndices(flags, b, dim, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (is_3d && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    flags(i, j, k, b) = TypeObstacle;
  } else {
    flags(i, j, k, b) = TypeFluid;
  }
}

static int tfluids_CudaMain_emptyDomain(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 2));
  const int32_t bnd = static_cast<int32_t>(lua_tointeger(L, 3));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  // Looped version - Actually not really any faster..
  // LaunchKernelLoop(L, &emptyDomainLoop, flags,
  //                  flags, is_3d, bnd, flags.nbatch(), flags.zsize(),
  //                  flags.ysize(), flags.xsize(), flags.numel());
  LaunchKernel(L, &emptyDomain, flags,
               flags, is_3d, bnd);
  return 0;
}

// *****************************************************************************
// flagsToOccupancy
// *****************************************************************************

__global__ void flagsToOccupancy(CudaFlagGrid flags,
                                 CudaFlagGrid occupancy) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  float val;
  if (flags.isFluid(i, j, k, b)) {
    val = 0;
  } else if (flags.isObstacle(i, j, k, b)) {
    val = 1;
  } else {
    val = -1;  // Can't throw error in kernel. Set to -1 and check min.
  }
  occupancy(i, j, k, b) = val;
}

static int tfluids_CudaMain_flagsToOccupancy(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_occupancy = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));

  // Normally, we would pass this in, but actually it doesn't make a difference
  // to the calculation.
  const bool is_3d = tensor_flags->size[2] > 1;

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaFlagGrid occupancy = toCudaFlagGrid(state, tensor_occupancy, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &flagsToOccupancy, flags,
               flags, occupancy);

  // We could be pedantic and check that the occupancy grid is OK. But this
  // reduction is very expensive on GPU.
  // if (THCudaTensor_minall(state, tensor_occupancy) < 0) {
  //   luaL_error(L, "ERROR: unsupported flag cell found!");
  // } 

  return 0;
}

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

__global__ void velocityUpdateForward(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid pressure,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta doesn't touch the velocity on the boundaries (i.e.
    // it stays constant).
    return;
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
    if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
      vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                             pressure(i, j, k - 1, b));
    }

    if (flags.isEmpty(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) -= pressure(i, j, k, b);
    }
    if (flags.isEmpty(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) -= pressure(i, j, k, b);
    }
    if (flags.is_3d() && flags.isEmpty(i, j, k - 1, b)) {
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
    if (flags.is_3d()) {
      if (flags.isFluid(i, j, k - 1, b)) {
        vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
      } else {
        vel(i, j, k, 2, b)  = 0.f;
      }
    }
  }
}

static int tfluids_CudaMain_velocityUpdateForward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid pressure = toCudaRealGrid(state, tensor_p, is_3d);
 
  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &velocityUpdateForward, flags,
               flags, vel, pressure, bnd);
  
  return 0;  // Recall: number of return values on the lua stack. 
}


// *****************************************************************************
// velocityUpdateBackward
// *****************************************************************************

__global__ void velocityUpdateBackward(
    CudaFlagGrid flags, CudaMACGrid grad_output, CudaRealGrid grad_p,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border in the forward pass, so they do
    // not contribute gradient.
    return;
  }

  const CudaVec3 go(grad_output(i, j, k, b));

  // TODO(tompson): I'm sure these atomic add calls are slow! We should
  // probably change this from a scatter to a gather op to avoid having to use
  // them at all.
  // (NVIDIA state that atomic operations on global memory are extremely slow)
  // but on shared memory it is OK. So we could copy to shared first, use
  // atomic ops there then use a small number of atomic ops back to global mem
  // (probably rewriting it as a gather would be easier).
  if (flags.isFluid(i, j, k, b)) {
    if (flags.isFluid(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.x);
      atomicAdd(&grad_p(i - 1, j, k, b), go.x);
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.y);
      atomicAdd(&grad_p(i, j - 1, k, b), go.y);
    }
    if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.z); 
      atomicAdd(&grad_p(i, j, k - 1, b), go.z);
    }

    if (flags.isEmpty(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.x);
    }
    if (flags.isEmpty(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.y);
    }
    if (flags.is_3d() && flags.isEmpty(i, j, k - 1, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.z);
    }
  }
  else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
    // don't change velocities in outflow cells   
    if (flags.isFluid(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i - 1, j, k, b), go.x);
    } else {
      // Output doesn't depend on p, so gradient is zero and so doesn't
      // contribute.
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j - 1, k, b), go.y);
    } else {
      // Output doesn't depend on p, so gradient is zero and so doesn't
      // contribute.
    }
    if (flags.is_3d()) {
      if (flags.isFluid(i, j, k - 1, b)) {
        atomicAdd(&grad_p(i, j, k - 1, b), go.z);
      } else {
        // Output doesn't depend on p, so gradient is zero and so
        // doesn't contribute.
      }
    }
  }
}

static int tfluids_CudaMain_velocityUpdateBackward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  THCudaTensor* tensor_grad_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid grad_output = toCudaMACGrid(state, tensor_grad_output, is_3d);
  CudaRealGrid grad_p = toCudaRealGrid(state, tensor_grad_p, is_3d);

  // Firstly, we're going to accumulate gradient contributions, so set
  // grad_p to 0.
  THCudaTensor_zero(state, tensor_grad_p);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &velocityUpdateBackward, flags,
               flags, grad_output, grad_p, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// addBuoyancy
// *****************************************************************************

__global__ void addBuoyancy(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid factor,
    THCDeviceTensor<float, 1> strength, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  } 

  if (!flags.isFluid(i, j, k, b)) {
    return;
  }
  if (flags.isFluid(i - 1, j, k, b)) {
    vel(i, j, k, 0, b) += (0.5f * strength[0] *
                           (factor(i, j, k, b) + factor(i - 1, j, k, b)));
  }
  if (flags.isFluid(i, j - 1, k, b)) {
    vel(i, j, k, 1, b) += (0.5f * strength[1] *
                           (factor(i, j, k, b) + factor(i, j - 1, k, b)));
  }
  if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
    vel(i, j, k, 2, b) += (0.5f * strength[2] *
                           (factor(i, j, k, b) + factor(i, j, k - 1, b)));
  }

}
static int tfluids_CudaMain_addBuoyancy(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_density = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_gravity = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_strength = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  const float dt = static_cast<float>(lua_tonumber(L, 6));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));

  if (tensor_gravity->nDimension != 1 || tensor_gravity->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }
  if (tensor_strength->nDimension != 1 || tensor_strength->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid factor = toCudaRealGrid(state, tensor_density, is_3d);

  THCudaTensor_copy(state, tensor_strength, tensor_gravity);
  THCudaTensor_mul(state, tensor_strength, tensor_strength,
                   -1.0f * dt / flags.getDx());
  THCDeviceTensor<float, 1> dev_strength =
      toDeviceTensor<float, 1>(state, tensor_strength);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &addBuoyancy, flags,
               flags, vel, factor, dev_strength, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// vorticityConfinement
// *****************************************************************************

__global__ void AddForceField(
    CudaFlagGrid flags, CudaMACGrid vel, CudaVecGrid force, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  const bool curFluid = flags.isFluid(i, j, k, b);
  const bool curEmpty = flags.isEmpty(i, j, k, b);
  if (!curFluid && !curEmpty) {
    return;
  }

  if (flags.isFluid(i - 1, j, k, b) || 
      (curFluid && flags.isEmpty(i - 1, j, k, b))) {
    vel(i, j, k, 0, b) += (0.5f *
                        (force(i - 1, j, k, 0, b) + force(i, j, k, 0, b)));
  }

  if (flags.isFluid(i, j - 1, k, b) ||
      (curFluid && flags.isEmpty(i, j - 1, k, b))) {
    vel(i, j, k, 1, b) += (0.5f * 
                        (force(i, j - 1, k, 1, b) + force(i, j, k, 1, b)));
  }

  if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
      (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
    vel(i, j, k, 2, b) += (0.5f *
                        (force(i, j, k - 1, 2, b) + force(i, j, k, 2, b)));
  }
}

__global__ void GetCentered(CudaFlagGrid flags, CudaMACGrid vel,
                            CudaVecGrid centered, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    centered.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }
  centered.setSafe(i, j, k, b, vel.getCentered(i, j, k, b));
}

__global__ void GetCurlAndCurlNorm(
    CudaFlagGrid flags, CudaVecGrid centered, CudaVecGrid curl,
    CudaRealGrid curl_norm, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    curl.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    curl_norm(i, j, k, b) = 0;
    return;
  }
  const CudaVec3 cur_curl(centered.curl(i, j, k, b));
  curl.setSafe(i, j, k, b, cur_curl);
  curl_norm(i, j, k, b) = cur_curl.norm();
}

__global__ void GetVorticityConfinementForce(
    CudaFlagGrid flags, CudaVecGrid curl, CudaRealGrid curl_norm,
    const float strength, CudaVecGrid force, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd || 
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Don't add force on the boundaries.
    force.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }

  CudaVec3 grad(0, 0, 0);
  grad.x = 0.5f * (curl_norm(i + 1, j, k, b) - curl_norm(i - 1, j, k, b));
  grad.y = 0.5f * (curl_norm(i, j + 1, k, b) - curl_norm(i, j - 1, k, b));
  if (flags.is_3d()) {
    grad.z = 0.5f * (curl_norm(i, j, k + 1, b) - curl_norm(i, j, k - 1, b));
  }
  grad.normalize();
  
  force.setSafe(i, j, k, b, CudaVec3::cross(grad, curl(i, j, k, b)) * strength);
}

static int tfluids_CudaMain_vorticityConfinement(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  const float strength = static_cast<float>(lua_tonumber(L, 3));
  THCudaTensor* tensor_centered = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_curl = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* tensor_curl_norm = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));
  THCudaTensor* tensor_force = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 7, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 8));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaVecGrid centered = toCudaVecGrid(state, tensor_centered, is_3d);
  CudaVecGrid curl = toCudaVecGrid(state, tensor_curl, true);  // Always 3D.
  CudaRealGrid curl_norm = toCudaRealGrid(state, tensor_curl_norm, is_3d);
  CudaVecGrid force = toCudaVecGrid(state, tensor_force, is_3d);

  // First calculate the centered velocity.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &GetCentered, flags,
               flags, vel, centered, bnd);

  // Now calculate the curl and it's (l2) norm (of the centered velocities).
  LaunchKernel(L, &GetCurlAndCurlNorm, flags,
               flags, centered, curl, curl_norm, bnd);

 
  // Now calculate the vorticity confinement force.
  LaunchKernel(L, &GetVorticityConfinementForce, flags,
               flags, curl, curl_norm, strength, force, bnd);

  // Now apply the force.
  LaunchKernel(L, &AddForceField, flags,
               flags, vel, force, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// volumetricUpsamplingNearestForward
// *****************************************************************************

__global__ void volumetricUpSamplingNearestForward(
    const int ratio, THCDeviceTensor<float, 5> in,
    THCDeviceTensor<float, 5> out) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int chan = blockIdx.y;
  const int batch = blockIdx.z;
  if (pnt_id >= (out.getSize(2) * out.getSize(3) * out.getSize(4))) {
    return;
  }
  const int x = pnt_id % out.getSize(4);
  const int y = (pnt_id / out.getSize(4)) % out.getSize(3);
  const int z = pnt_id / (out.getSize(3) * out.getSize(4));

  const int xin = x / ratio;
  const int yin = y / ratio;
  const int zin = z / ratio;
  const float inVal = in[batch][chan][zin][yin][xin];
  out[batch][chan][z][y][x] = inVal;
}

static int tfluids_CudaMain_volumetricUpSamplingNearestForward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THCudaTensor* input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));

  if (input->nDimension != 5 || output->nDimension != 5) {
    luaL_error(L, "ERROR: input and output must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (output->size[0] != nbatch || output->size[1] != nfeat ||
      output->size[2] != zdim * ratio || output->size[3] != ydim * ratio ||
      output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : output size mismatch.");
  }

  THCDeviceTensor<float, 5> dev_in = toDeviceTensor<float, 5>(state, input);
  THCDeviceTensor<float, 5> dev_out = toDeviceTensor<float, 5>(state, output);

  if (!THCudaTensor_isContiguous(state, input)) {
    luaL_error(L, "ERROR: input must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, output)) {
    luaL_error(L, "ERROR: output must be contiguous");
  }

  // One thread per output element.
  int nplane = dev_out.getSize(2) * dev_out.getSize(3) * dev_out.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), dev_out.getSize(1),
                 dev_out.getSize(0));
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  volumetricUpSamplingNearestForward<<<grid_size, block_size, 0,
                                       THCState_getCurrentStream(state)>>>(
      ratio, dev_in, dev_out);

  return 0;
}

// *****************************************************************************
// volumetricUpsamplingNearestBackward
// *****************************************************************************

__global__ void volumetricUpSamplingNearestBackward(
    const int ratio, THCDeviceTensor<float, 5> grad_out,
    THCDeviceTensor<float, 5> grad_in) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int chan = blockIdx.y;
  const int batch = blockIdx.z; 
  if (pnt_id >= (grad_in.getSize(2) * grad_in.getSize(3) *
      grad_in.getSize(4))) {
    return;
  }
  const int x = pnt_id % grad_in.getSize(4);
  const int y = (pnt_id / grad_in.getSize(4)) % grad_in.getSize(3);
  const int z = pnt_id / (grad_in.getSize(3) * grad_in.getSize(4));
 
  float sum = 0.0f;

  // Now accumulate gradients from the upsampling window.
  for (int32_t zup = 0; zup < ratio; zup++) { 
    for (int32_t yup = 0; yup < ratio; yup++) { 
      for (int32_t xup = 0; xup < ratio; xup++) {
        const int xin = x * ratio + xup;
        const int yin = y * ratio + yup;
        const int zin = z * ratio + zup;
        const float val = grad_out[batch][chan][zin][yin][xin];
        sum += val;
      }
    }
  }
        
  grad_in[batch][chan][z][y][x] = sum;
}

static int tfluids_CudaMain_volumetricUpSamplingNearestBackward(lua_State *L) {
  THCState* state = cutorch_getstate(L);
  
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THCudaTensor* input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* grad_input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  
  if (input->nDimension != 5 || grad_output->nDimension != 5 ||
      grad_input->nDimension != 5) {
    luaL_error(L, "ERROR: input, gradOutput and gradInput must be dim 5");
  }
  
  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (grad_output->size[0] != nbatch || grad_output->size[1] != nfeat ||
      grad_output->size[2] != zdim * ratio ||
      grad_output->size[3] != ydim * ratio ||
      grad_output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : gradOutput size mismatch.");
  }

  if (grad_input->size[0] != nbatch || grad_input->size[1] != nfeat ||
      grad_input->size[2] != zdim || grad_input->size[3] != ydim ||
      grad_input->size[4] != xdim) {
    luaL_error(L, "ERROR: input : gradInput size mismatch.");
  }

  THCDeviceTensor<float, 5> dev_in = toDeviceTensor<float, 5>(state, input);
  THCDeviceTensor<float, 5> dev_grad_out = toDeviceTensor<float, 5>(
      state, grad_output);
  THCDeviceTensor<float, 5> dev_grad_in = toDeviceTensor<float, 5>(
    state, grad_input);
  
  if (!THCudaTensor_isContiguous(state, input)) {
    luaL_error(L, "ERROR: input must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, grad_output)) {
    luaL_error(L, "ERROR: gradOutput must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, grad_input)) {
    luaL_error(L, "ERROR: gradInput must be contiguous");
  }

  // One thread per grad_input element.
  // TODO(tompson): This is slow. Switch to a looping kernel.
  int nplane = dev_grad_in.getSize(2) * dev_grad_in.getSize(3) *
    dev_grad_in.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), dev_grad_in.getSize(1),
                 dev_grad_in.getSize(0));  
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);
  
  volumetricUpSamplingNearestBackward<<<grid_size, block_size, 0,
                                        THCState_getCurrentStream(state)>>>(
      ratio, dev_grad_out, dev_grad_in);

  return 0;
}

// *****************************************************************************
// signedDistanceField
// *****************************************************************************

__global__ void signedDistanceField(
    CudaFlagGrid flags, const int search_rad, CudaRealGrid dst) {
  int b, chan, z, y, x;
  if (GetKernelIndices(flags, b, chan, z, y, x)) {
    return;
  }

  if (flags.isObstacle(x, y, z, b)) {
    dst(x, y, z, b) = 0;
  }

  float dist_sq = static_cast<float>(search_rad * search_rad);
  const int zmin = max(0, z - search_rad);;
  const int zmax = min((int)flags.zsize() - 1, z + search_rad);
  const int ymin = max(0, y - search_rad);;
  const int ymax = min((int)flags.ysize() - 1, y + search_rad);
  const int xmin = max(0, x - search_rad);;
  const int xmax = min((int)flags.xsize() - 1, x + search_rad);
  for (int zsearch = zmin; zsearch <= zmax; zsearch++) {
    for (int ysearch = ymin; ysearch <= ymax; ysearch++) {
      for (int xsearch = xmin; xsearch <= xmax; xsearch++) {
        if (flags.isObstacle(xsearch, ysearch, zsearch, b)) {
          const float cur_dist_sq = ((z - zsearch) * (z - zsearch) +
                                     (y - ysearch) * (y - ysearch) +
                                     (x - xsearch) * (x - xsearch));
          if (dist_sq > cur_dist_sq) {
            dist_sq = cur_dist_sq;
          }
        }
      }
    }
  }
  dst(x, y, z, b) = sqrt(dist_sq);
}

static int tfluids_CudaMain_signedDistanceField(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  const int32_t search_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THCudaTensor* tensor_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaRealGrid dst = toCudaRealGrid(state, tensor_dst, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &signedDistanceField, flags,
               flags, search_rad, dst);

  return 0;
}

//******************************************************************************
// solveLinearSystemPCG
//******************************************************************************

// solveLinearSystemPCG lua entry point.
static int tfluids_CudaMain_solveLinearSystemPCG(lua_State *L) {
  init_cusparse();  // No op if already initialized.
  luaL_error(L, "ERROR: solveLinearSystemPCG not implemented");  // DONOTSUBMIT

  return 0;
}

//******************************************************************************
// INIT METHODS
//******************************************************************************
static const struct luaL_Reg tfluids_CudaMain__ [] = {
  {"advectScalar", tfluids_CudaMain_advectScalar},
  {"advectVel", tfluids_CudaMain_advectVel},
  {"setWallBcsForward", tfluids_CudaMain_setWallBcsForward},
  {"vorticityConfinement", tfluids_CudaMain_vorticityConfinement},
  {"addBuoyancy", tfluids_CudaMain_addBuoyancy},
  {"velocityUpdateForward", tfluids_CudaMain_velocityUpdateForward},
  {"velocityUpdateBackward", tfluids_CudaMain_velocityUpdateBackward},
  {"velocityDivergenceForward", tfluids_CudaMain_velocityDivergenceForward},
  {"velocityDivergenceBackward", tfluids_CudaMain_velocityDivergenceBackward},
  {"emptyDomain", tfluids_CudaMain_emptyDomain},
  {"flagsToOccupancy", tfluids_CudaMain_flagsToOccupancy},
  {"solveLinearSystemPCG", tfluids_CudaMain_solveLinearSystemPCG},
  {"volumetricUpSamplingNearestForward",
   tfluids_CudaMain_volumetricUpSamplingNearestForward},
  {"volumetricUpSamplingNearestBackward",
   tfluids_CudaMain_volumetricUpSamplingNearestBackward},
  {"signedDistanceField", tfluids_CudaMain_signedDistanceField},
  {NULL, NULL}  // NOLINT
};

const struct luaL_Reg* tfluids_CudaMain_getMethodsTable() {
  return tfluids_CudaMain__;
}

void tfluids_CudaMain_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, tfluids_CudaMain__, "tfluids");
}
