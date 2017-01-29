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

// This is a very, very barebones replication of the Manta grids. It's just
// so that we can more easily transfer KERNEL functions across.

#include <iostream>
#include <sstream>
#include <mutex>

class tfluids_(GridBase) {
public:
  // Note: tensors (grid) passed to GridBase will remain owned by the caller.
  // The caller is expected to make sure the pointers remain valid while
  // the GridBase instance is used (and to call THTensor_(free) if required).
  // TODO(tompson): This class should really be pure virtual.
  explicit tfluids_(GridBase)(THTensor* grid, bool is_3d) :
      is_3d_(is_3d), tensor_(grid), p_grid_(THTensor_(data)(grid)) {
    if (grid->nDimension != 5) {
      THError("GridBase: dim must be 5D (even if simulation is 2D).");
    }

    if (!is_3d_ && zsize() != 1) {
      THError("GridBase: 2D grid must have zsize == 1.");
    }
  }

  inline int32_t nbatch() const { return tensor_->size[0]; }
  inline int32_t nchan() const { return tensor_->size[1]; }
  inline int32_t zsize() const { return tensor_->size[2]; }
  inline int32_t ysize() const { return tensor_->size[3]; }
  inline int32_t xsize() const { return tensor_->size[4]; }

  inline int32_t bstride() const { return tensor_->stride[0]; }
  inline int32_t cstride() const { return tensor_->stride[1]; }
  inline int32_t zstride() const { return tensor_->stride[2]; }
  inline int32_t ystride() const { return tensor_->stride[3]; }
  inline int32_t xstride() const { return tensor_->stride[4]; }

  inline real getDx() const {
    const int32_t size_max = std::max(xsize(), std::max(ysize(), zsize()));
    return static_cast<real>(1) / static_cast<real>(size_max);
  }

  inline bool is_3d() const { return is_3d_; }

  inline Int3 getSize() const { return Int3(xsize(), ysize(), zsize()); }

  inline bool isInBounds(const Int3& p, int bnd) const {
    bool ret = (p.x >= bnd && p.y >= bnd && p.x < xsize() - bnd &&
                p.y < ysize() - bnd);
    if (is_3d_) {
      ret &= (p.z >= bnd && p.z < zsize() - bnd);
    } else {
      ret &= (p.z == 0);
    }
    return ret; 
  }

  inline bool isInBounds(const tfluids_(vec3)& p, int bnd) const {
    return isInBounds(toInt3(p), bnd);
  }

private:
  // Note: Child classes should use getters!
  THTensor* const tensor_;
  real* const p_grid_;  // The actual flat storage.
  const bool is_3d_;
  static std::mutex mutex_;

  // The indices i, j, k, c, b are x, y, z, chan and batch respectively.
  inline int32_t index5d(int32_t i, int32_t j, int32_t k, int32_t c,
                         int32_t b) const {
    if (i >= xsize() || j >= ysize() || k >= zsize() || c >= nchan() ||
        b >= nbatch() || i < 0 || j < 0 || k < 0 || c < 0 || b < 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      std::stringstream ss;
      ss << "GridBase: index4d out of bounds:" << std::endl
         << "  (i, j, k, c, b) = (" << i << ", " << j
         << ", " << k << ", " << c << ", " << b << "), size = (" << xsize()
         << ", " << ysize() << ", " << zsize() << ", " << nchan() 
         << nbatch() << ")";
      std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
      PrintStacktrace();
      std::cerr << std::endl;
      THError("GridBase: index4d out of bounds");
      return 0;
    }
    return (i * xstride() + j * ystride() + k * zstride() + c * cstride() +
            b * bstride());
  }

protected:
  // Use operator() methods in child classes to get at data.
  // Note: if the storage is offset (i.e. because we've selected along the
  // batch dim), this is taken care of in THTensor_(data) (i.e. it returns
  // self->storage->data + self->storageOffset).
  inline real& data(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return p_grid_[index5d(i, j, k, c, b)];
  }
  inline real data(int32_t i, int32_t j, int32_t k, int32_t c,
                   int32_t b) const {
    return p_grid_[index5d(i, j, k, c, b)];
  }
  // Build index is used in interpol and interpolComponent. It replicates
  // the BUILD_INDEX macro in Manta's util/interpol.h.
  inline void buildIndex(int32_t& xi, int32_t& yi, int32_t& zi,
                         real& s0, real& t0, real& f0,
                         real& s1, real& t1, real& f1,
                         const tfluids_(vec3)& pos) const {
    const real px = pos.x - static_cast<real>(0.5);
    const real py = pos.y - static_cast<real>(0.5);
    const real pz = pos.z - static_cast<real>(0.5);
    xi = static_cast<int32_t>(px);
    yi = static_cast<int32_t>(py);
    zi = static_cast<int32_t>(pz);
    s1 = px - static_cast<real>(xi);
    s0 = static_cast<real>(1) - s1;
    t1 = py - static_cast<real>(yi);
    t0 = static_cast<real>(1) - t1;
    f1 = pz - static_cast<real>(zi);
    f0 = static_cast<real>(1) - f1;
    // Clamp to border.
    if (px < static_cast<real>(0)) {
      xi = 0;
      s0 = static_cast<real>(1);
      s1 = static_cast<real>(0);
    }
    if (py < static_cast<real>(0)) {
      yi = 0;
      t0 = static_cast<real>(1);
      t1 = static_cast<real>(0);
    }
    if (pz < static_cast<real>(0)) {
      zi = 0;
      f0 = static_cast<real>(1);
      f1 = static_cast<real>(0);
    }
    if (xi >= xsize() - 1) {
      xi = xsize() - 2;
      s0 = static_cast<real>(0);
      s1 = static_cast<real>(1);
    }
    if (yi >= ysize() - 1) {
      yi = ysize() - 2;
      t0 = static_cast<real>(0);
      t1 = static_cast<real>(1);
    }
    if (zsize() > 1) {
      if (zi >= zsize() - 1) {
        zi = zsize() - 2;
        f0 = static_cast<real>(0);
        f1 = static_cast<real>(1);
      }
    }
  }
};

std::mutex tfluids_(GridBase)::mutex_;

class tfluids_(FlagGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(FlagGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 1) {
      THError("FlagGrid: nchan must be 1 (scalar).");
    }
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }
  
  inline real operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  }

  inline bool isFluid(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeFluid;
  }

  inline bool isObstacle(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeObstacle;
  }

  inline bool isObstacle(const Int3& pos, int32_t b) const {
    return isObstacle(pos.x, pos.y, pos.z, b);
  }

  inline bool isStick(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeStick;
  }

  inline bool isEmpty(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeEmpty;
  }

  inline bool isOutflow(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeOutflow;
  }
};

// Our RealGrid is supposed to be like Grid<Real> in Manta.
class tfluids_(RealGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(RealGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 1) {
      THError("RealGrid: nchan must be 1 (scalar).");
    }
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }

  inline real operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  };

  inline real getInterpolatedHi(const tfluids_(vec3)& pos, int32_t order,
                                int32_t b) const {
    switch (order) {
    case 1:
      return interpol(pos, b);
    case 2:
      THError("getInterpolatedHi ERROR: cubic not supported.");
      // TODO(tompson): implement this.
      break;
    default:
      THError("getInterpolatedHi ERROR: order not supported.");
      break;
    }
    return 0;
  }

  inline real interpol(const tfluids_(vec3)& pos, int32_t b) const {
    int32_t xi, yi, zi;
    real s0, t0, f0, s1, t1, f1;
    buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos); 

    if (is_3d()) {
      return ((data(xi, yi, zi, 0, b) * t0 +
               data(xi, yi + 1, zi, 0, b) * t1) * s0 
          + (data(xi + 1, yi, zi, 0, b) * t0 +
             data(xi + 1, yi + 1, zi, 0, b) * t1) * s1) * f0
          + ((data(xi, yi, zi + 1, 0, b) * t0 +
             data(xi, yi + 1, zi + 1, 0, b) * t1) * s0
          + (data(xi + 1, yi, zi + 1, 0, b) * t0 +
             data(xi + 1, yi + 1, zi + 1, 0, b) * t1) * s1) * f1;
    } else {
       return ((data(xi, yi, 0, 0, b) * t0 +
                data(xi, yi + 1, 0, 0, b) * t1) * s0
          + (data(xi + 1, yi, 0, 0, b) * t0 +
             data(xi + 1, yi + 1, 0, 0, b) * t1) * s1);
    }
  }
};

class tfluids_(MACGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(MACGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 2 && nchan() != 3) {
      THError("MACGrid: input tensor size[0] is not 2 or 3");
    }
    if (!is_3d && zsize() != 1) {
      THError("MACGrid: 2D tensor does not have zsize == 1");
    }
  }

  // Note: as per other functions, we DO NOT bounds check getCentered. You must
  // not call this method on the edge of the simulation domain.
  const tfluids_(vec3) getCentered(int32_t i, int32_t j, int32_t k,
                                   int32_t b) const {  
    const real x = static_cast<real>(0.5) * (data(i, j, k, 0, b) +
                                             data(i + 1, j, k, 0, b));
    const real y = static_cast<real>(0.5) * (data(i, j, k, 1, b) +
                                             data(i, j + 1, k, 1, b));
    const real z = !is_3d() ? static_cast<real>(0) :
        static_cast<real>(0.5) * (data(i, j, k, 2, b) +
                                  data(i, j, k + 1, 2, b));
    return tfluids_(vec3)(x, y, z);
  }

  inline const tfluids_(vec3) operator()(int32_t i, int32_t j,
                                         int32_t k, int32_t b) const {
    tfluids_(vec3) ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<real>(0) : data(i, j, k, 2, b);
    return ret;
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k, int32_t c,
                          int32_t b) {
    return data(i, j, k, c, b);
  }

  inline real operator()(int32_t i, int32_t j, int32_t k, int32_t c,
                         int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // MACGrid is 2D.
  inline void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                      const tfluids_(vec3)& val) {
    data(i, j, k, 0, b) = val.x;
    data(i, j, k, 1, b) = val.y;
    if (is_3d()) {
      data(i, j, k, 2, b) = val.z;
    } else {
      // This is a pedantic sanity check. We shouldn't be trying to set the
      // z component on a 2D MAC Grid with anything but zero. This is to make
      // sure that the end user fully understands what this function does.
      if (val.z != 0) {
        THError("MACGrid: setSafe z-component is non-zero for a 2D grid.");
      }
    }
  }

  inline tfluids_(vec3) getAtMACX(int32_t i, int32_t j, int32_t k,
                                  int32_t b) const {
    tfluids_(vec3) v;
    v.x = data(i, j, k, 0, b);
    v.y = (real)0.25 * (data(i, j, k, 1, b) + data(i - 1, j, k, 1, b) +
                        data(i, j + 1, k, 1, b) + data(i - 1, j + 1, k, 1, b));
    if (is_3d()) {
      v.z = (real)0.25* (data(i, j, k, 2, b) + data(i - 1, j, k, 2, b) +
                         data(i, j, k + 1, 2, b) + data(i - 1, j, k + 1, 2, b));
    } else {
      v.z = (real)0;
    }
    return v;
  }

  inline tfluids_(vec3) getAtMACY(int32_t i, int32_t j, int32_t k,
                                  int32_t b) const {
    tfluids_(vec3) v;
    v.x = (real)0.25 * (data(i, j, k, 0, b) + data(i, j - 1, k, 0, b) +
                        data(i + 1, j, k, 0, b) + data(i + 1, j - 1, k, 0, b));
    v.y = data(i, j, k, 1, b);
    if (is_3d()) {
      v.z = (real)0.25* (data(i, j, k, 2, b) + data(i, j - 1, k, 2, b) +
                         data(i, j, k + 1, 2, b) + data(i, j - 1, k + 1, 2, b));
    } else { 
      v.z = (real)0;
    }
    return v;
  }

  inline tfluids_(vec3) getAtMACZ(int32_t i, int32_t j, int32_t k,
                                  int32_t b) const {
    tfluids_(vec3) v;
    v.x = (real)0.25 * (data(i, j, k, 0, b) + data(i, j, k - 1, 0, b) +
                        data(i + 1, j, k, 0, b) + data(i + 1, j, k - 1, 0, b));
    v.y = (real)0.25 * (data(i, j, k, 1, b) + data(i, j, k - 1, 1, b) +
                        data(i, j + 1, k, 1, b) + data(i, j + 1, k - 1, 1, b));
    if (is_3d()) {
      v.z = data(i, j, k, 2, b);
    } else {
      v.z = (real)0;
    }
    return v;
  }

  template <int comp>
  inline real getInterpolatedComponentHi(const tfluids_(vec3)& pos,
                                         int32_t order, int32_t b) const {
    switch (order) {
    case 1:
      return interpolComponent<comp>(pos, b);
    case 2:
      THError("getInterpolatedComponentHi ERROR: cubic not supported.");
      // TODO(tompson): implement this.
      break;
    default:
      THError("getInterpolatedComponentHi ERROR: order not supported.");
      break;
    }
    return 0;
  }

  template <int c>
  inline real interpolComponent(const tfluids_(vec3)& pos, int32_t b) const {
    int32_t xi, yi, zi;
    real s0, t0, f0, s1, t1, f1;
    buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos);

    if (is_3d()) {
      return ((data(xi, yi, zi, c, b) * t0 +
               data(xi, yi + 1, zi, c, b) * t1) * s0
          + (data(xi + 1, yi, zi, c, b) * t0 +
             data(xi + 1, yi + 1, zi, c, b) * t1) * s1) * f0
          + ((data(xi, yi, zi + 1, c, b) * t0 +
              data(xi, yi + 1, zi + 1, c, b) * t1) * s0
          + (data(xi + 1, yi, zi + 1, c, b) * t0 +
             data(xi + 1, yi + 1, zi + 1, c, b) * t1) * s1) * f1;
    } else {
       return ((data(xi, yi, 0, c, b) * t0 +
                data(xi, yi + 1, 0, c, b) * t1) * s0
          + (data(xi + 1, yi, 0, c, b) * t0 +
             data(xi + 1, yi + 1, 0, c, b) * t1) * s1);
    }
  }
};

class tfluids_(VecGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(VecGrid)(THTensor* grid, bool is_3d) :
      tfluids_(GridBase)(grid, is_3d) {
    if (nchan() != 2 && nchan() != 3) {
      THError("VecGrid: input tensor size[0] is not 2 or 3");
    }
    if (!is_3d && zsize() != 1) {
      THError("VecGrid: 2D tensor does not have zsize == 1");
    }
  }

  inline const tfluids_(vec3) operator()(int32_t i, int32_t j,
                                         int32_t k, int32_t b) const {
    tfluids_(vec3) ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<real>(0) : data(i, j, k, 2, b);
    return ret;
  }

  inline real& operator()(int32_t i, int32_t j, int32_t k, int32_t c,
                          int32_t b) {
    return data(i, j, k, c, b);
  }

  inline real operator()(int32_t i, int32_t j, int32_t k, int32_t c,
                         int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // VecGrid is 2D.
  inline void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                      const tfluids_(vec3)& val) {
    data(i, j, k, 0, b) = val.x;
    data(i, j, k, 1, b) = val.y;
    if (is_3d()) {
      data(i, j, k, 2, b) = val.z;
    } else {
      // This is a pedantic sanity check. We shouldn't be trying to set the
      // z component on a 2D Vec Grid with anything but zero. This is to make
      // sure that the end user fully understands what this function does.
      if (val.z != 0) {
        THError("VecGrid: setSafe z-component is non-zero for a 2D grid.");
      }
    }
  }

  // Note: you CANNOT call curl on the border of the grid (if you do then
  // the data(...) calls will throw an error.
  // Also note that curl in 2D is a scalar, but we will return a vector anyway
  // with the scalar value being in the 3rd dim.
  inline tfluids_(vec3) curl(int32_t i, int32_t j, int32_t k, int32_t b) {
     tfluids_(vec3) v(0, 0, 0);
     v.z = static_cast<real>(0.5) * ((data(i + 1, j, k, 1, b) -
                                      data(i - 1, j, k, 1, b)) -
                                     (data(i, j + 1, k, 0, b) -
                                      data(i, j - 1, k, 0, b)));
    if(is_3d()) {
        v.x = static_cast<real>(0.5) * ((data(i, j + 1, k, 2, b) -
                                         data(i, j - 1, k, 2, b)) -
                                        (data(i, j, k + 1, 1, b) -
                                         data(i, j, k - 1, 1, b)));
        v.y = static_cast<real>(0.5) * ((data(i, j, k + 1, 0, b) -
                                         data(i, j, k - 1, 0, b)) -
                                        (data(i + 1, j, k, 2, b) -
                                         data(i - 1, j, k, 2, b)));
    }
    return v;
  }
};

