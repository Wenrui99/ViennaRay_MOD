#pragma once

#include <rayGeometry.hpp>

#include <limits>
#include <utility>
#include <vector>

namespace viennaray {

using namespace viennacore;

/// Geometry built from one sphere per point, e.g. one sphere per surface atom.
///
/// Unlike the disk geometry the spheres are solid bodies: Embree reports the
/// first intersection with the union of all spheres and that point of impact
/// always lies on the exposed part of the union surface. A ray therefore
/// deposits its weight on exactly one sphere and the trace kernel does not
/// spread it over the overlapping neighbors (which is what the infinitely thin
/// disks require). Consequently no point neighborhood is built here.
///
/// The area that turns the accumulated ray weight into a flux cannot be
/// derived from the radius once the spheres overlap, as they do in a crystal
/// lattice. It is therefore supplied from the outside, see
/// setUniformNormalizationArea() and setNormalizationAreas().
template <typename NumericType, int D>
class GeometrySphere : public Geometry<NumericType, D> {
public:
  GeometrySphere() : Geometry<NumericType, D>(GeometryType::SPHERE) {}

  /// Builds the Embree sphere geometry from the sphere centers. `radii` may be
  /// empty, in which case `defaultRadius` is used for every sphere.
  ///
  /// The bounding box encloses the sphere *centers* only, the same convention
  /// the disk geometry uses. Padding it by the radius is left to the caller,
  /// because the correct padding depends on the boundary conditions (see
  /// TraceSphere::apply).
  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, Dim>> const &points,
                    NumericType const defaultRadius,
                    std::vector<NumericType> const &radii = {}) {
    static_assert(Dim >= static_cast<size_t>(D),
                  "Point dimension must be >= D");
    assert((radii.empty() || radii.size() == points.size()) &&
           "GeometrySphere: points/radii size mismatch");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    this->pRtcGeometry_ = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    this->numPrimitives_ = static_cast<unsigned>(points.size());

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_4f_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), this->numPrimitives_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");

    for (int i = 0; i < D; ++i) {
      this->minCoords_[i] = std::numeric_limits<NumericType>::max();
      this->maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }
    if constexpr (D == 2) {
      this->minCoords_[2] = 0.;
      this->maxCoords_[2] = 0.;
    }

    maxRadius_ = 0;
    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      const NumericType radius = radii.empty() ? defaultRadius : radii[i];
      if (radius > maxRadius_)
        maxRadius_ = radius;

      pPointBuffer_[i].xx = static_cast<float>(points[i][0]);
      pPointBuffer_[i].yy = static_cast<float>(points[i][1]);
      pPointBuffer_[i].radius = static_cast<float>(radius);
      if constexpr (D == 2) {
        pPointBuffer_[i].zz = 0.f;
      } else {
        pPointBuffer_[i].zz = static_cast<float>(points[i][2]);
      }

      for (int d = 0; d < D; ++d) {
        if (points[i][d] < this->minCoords_[d])
          this->minCoords_[d] = points[i][d];
        if (points[i][d] > this->maxCoords_[d])
          this->maxCoords_[d] = points[i][d];
      }
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(this->pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(this->pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    if (this->materialIds_.size() != this->numPrimitives_) {
      this->materialIds_.resize(this->numPrimitives_, 0);
    }

    // The normalization areas belong to the previous geometry and would be
    // indexed with the new prim IDs otherwise.
    normalizationAreas_.clear();
    uniformArea_ = -1;
  }

  [[nodiscard]] Vec3D<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ &&
           "GeometrySphere: primID out of bounds");
    auto const &point = pPointBuffer_[primID];
    return Vec3D<NumericType>{static_cast<NumericType>(point.xx),
                              static_cast<NumericType>(point.yy),
                              static_cast<NumericType>(point.zz)};
  }

  [[nodiscard]] NumericType getRadius(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ &&
           "GeometrySphere: primID out of bounds");
    return static_cast<NumericType>(pPointBuffer_[primID].radius);
  }

  [[nodiscard]] NumericType getMaxRadius() const { return maxRadius_; }

  /// Same normalization area for every sphere. This is the right choice for a
  /// close packed atomistic surface: conservation of the particle current
  /// fixes the capture area of an atom on a flat opaque surface to the area
  /// per adsorption site, independently of how the spheres are shaped or how
  /// much they overlap. The resulting flux is a per site particle current.
  ///
  /// Has to be called *after* initGeometry().
  void setUniformNormalizationArea(NumericType area) {
    uniformArea_ = area;
    normalizationAreas_.clear();
  }

  /// One normalization area per sphere, e.g. the solvent accessible (exposed)
  /// area, if the flux is meant to be a surface flux density rather than a per
  /// site particle current.
  ///
  /// Has to be called *after* initGeometry().
  void setNormalizationAreas(std::vector<NumericType> areas) {
    normalizationAreas_ = std::move(areas);
    uniformArea_ = -1;
  }

  /// Area used by TraceSphere::normalizeFlux(). Without an explicit area this
  /// falls back to 2*pi*r^2, the capture cross section of a single *isolated*
  /// sphere, which is only meaningful for sparse geometries.
  [[nodiscard]] NumericType
  getNormalizationArea(const unsigned int primID) const {
    if (!normalizationAreas_.empty()) {
      assert(primID < normalizationAreas_.size() &&
             "GeometrySphere: primID out of bounds");
      return normalizationAreas_[primID];
    }
    if (uniformArea_ > 0)
      return uniformArea_;
    const NumericType radius = getRadius(primID);
    return static_cast<NumericType>(2. * M_PI) * radius * radius;
  }

  [[nodiscard]] bool hasNormalizationArea() const {
    return uniformArea_ > 0 || !normalizationAreas_.empty();
  }

  /// The normal of a sphere depends on the point of impact and is computed in
  /// the trace kernel; there is no per-primitive normal.
  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    return Vec3D<NumericType>{0, 0, 0};
  }

  /// Returns the sphere as (x, y, z, radius), as stored in the Embree vertex
  /// buffer. The trace kernel uses it to reconstruct the surface normal.
  std::array<float, 4> &getPrimRef(unsigned int primID) override {
    assert(primID < this->numPrimitives_ &&
           "GeometrySphere: primID out of bounds");
    return *reinterpret_cast<std::array<float, 4> *>(&pPointBuffer_[primID]);
  }

  bool checkGeometryEmpty() const override {
    return pPointBuffer_ == nullptr || this->pRtcGeometry_ == nullptr ||
           this->numPrimitives_ == 0;
  }

  void releaseGeometry() override {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (pPointBuffer_ == nullptr || this->pRtcGeometry_ == nullptr) {
      return;
    }
    rtcReleaseGeometry(this->pRtcGeometry_);
    pPointBuffer_ = nullptr;
    this->pRtcGeometry_ = nullptr;
  }

private:
  struct point_4f_t {
    // The sphere geometry uses a vertex buffer of x, y, z and radius in single
    // precision floating point types.
    float xx, yy, zz, radius;
  };
  point_4f_t *pPointBuffer_ = nullptr;

  NumericType maxRadius_ = 0;
  NumericType uniformArea_ = -1;
  std::vector<NumericType> normalizationAreas_;
};

} // namespace viennaray
