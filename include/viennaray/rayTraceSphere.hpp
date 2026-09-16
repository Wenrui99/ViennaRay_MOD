#pragma once

#include <rayGeometrySphere.hpp>
#include <rayTrace.hpp>

#include <vcLogger.hpp>

#include <algorithm>

namespace viennaray {

using namespace viennacore;

/// Ray tracer for a geometry made of one sphere per point, e.g. one sphere per
/// surface atom of an atomistic (KMC) surface.
///
/// The primitive ID reported to the particle is the index of the sphere, so a
/// per-primitive sticking probability (DiffuseParticleMultiSticking) is a per
/// *atom* sticking probability and the recorded flux maps back to the atom
/// without any intermediate surface mesh.
template <class NumericType, int D>
class TraceSphere final : public Trace<NumericType, D> {
  using TraceKernel =
      rayInternal::TraceKernel<NumericType, D, GeometryType::SPHERE>;

public:
  TraceSphere() = default;
  ~TraceSphere() override { geometry_.releaseGeometry(); }

  /// Run the ray tracer
  void apply() override {
    checkSettings();
    if (this->RTInfo_.error)
      return;

    // The geometry bounding box encloses the sphere centers.
    auto boundingBox = geometry_.getBoundingBox();
    auto traceSettings = rayInternal::getTraceSettings(this->sourceDirection_);

    // Lift the source plane clear of the spheres.
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, this->sourceDirection_, geometry_.getMaxRadius());

    // Lateral extent, i.e. the two directions carrying a boundary condition.
    // Neither of the two obvious choices is correct for a periodic lattice:
    // the centers alone shorten the period by one atomic spacing, padding them
    // by the radius lengthens it by 2r. The caller therefore has to supply the
    // exact period through setLateralBoundingBox(). Reflective and ignored
    // directions are padded so that the spheres stay inside the box.
    constexpr int numLateralDirs = D == 3 ? 2 : 1;
    for (int i = 0; i < numLateralDirs; ++i) {
      const int dir = traceSettings[1 + i];
      if (useLateralBox_) {
        boundingBox[0][dir] = lateralMin_[dir];
        boundingBox[1][dir] = lateralMax_[dir];
      } else {
        boundingBox[0][dir] -= geometry_.getMaxRadius();
        boundingBox[1][dir] += geometry_.getMaxRadius();
      }
    }

    auto boundary = Boundary<NumericType, D>(
        this->device_, boundingBox, this->boundaryConditions_, traceSettings);

    this->prepareSource(geometry_.getNumPrimitives(), boundingBox,
                        traceSettings);
    this->prepareLocalData(geometry_.getNumPrimitives());

    TraceKernel tracer(this->device_, geometry_, boundary, this->pSource_,
                       this->pParticle_, this->config_, this->dataLog_,
                       this->RTInfo_);
    tracer.setTracingData(&this->localData_, this->pGlobalData_.get());
    tracer.apply();
    ++this->config_.runNumber;

    boundary.releaseGeometry();
  }

  /// Set the ray tracing geometry: one sphere of radius `radius` per point.
  /// An individual radius per sphere can be passed in `radii`.
  ///
  /// Resets the normalization area, so call setUniformNormalizationArea() or
  /// setNormalizationAreas() afterwards.
  template <size_t Dim>
  void setGeometry(std::vector<VectorType<NumericType, Dim>> const &points,
                   const NumericType radius,
                   std::vector<NumericType> const &radii = {}) {
    this->gridDelta_ = radius;
    geometry_.initGeometry(this->device_, points, radius, radii);
  }

  /// Set the exact extent of the two boundary planes, i.e. of the directions
  /// perpendicular to the source direction. Required for periodic boundaries,
  /// where the planes have to sit exactly one lattice period apart. Only the
  /// lateral components of the two vectors are read.
  void setLateralBoundingBox(Vec3D<NumericType> const &min,
                             Vec3D<NumericType> const &max) {
    lateralMin_ = min;
    lateralMax_ = max;
    useLateralBox_ = true;
  }

  void resetLateralBoundingBox() { useLateralBox_ = false; }

  /// Set material ID's for each sphere.
  /// If not set, all material IDs are default 0.
  template <typename T> void setMaterialIds(std::vector<T> const &materialIds) {
    geometry_.setMaterialIds(materialIds);
  }

  /// Area used to turn the accumulated ray weight of a sphere into a flux, see
  /// GeometrySphere::setUniformNormalizationArea(). Has to be called after
  /// setGeometry().
  void setUniformNormalizationArea(const NumericType area) {
    geometry_.setUniformNormalizationArea(area);
  }

  /// One normalization area per sphere, see
  /// GeometrySphere::setNormalizationAreas(). Has to be called after
  /// setGeometry().
  void setNormalizationAreas(std::vector<NumericType> areas) {
    geometry_.setNormalizationAreas(std::move(areas));
  }

  [[nodiscard]] GeometrySphere<NumericType, D> const &getGeometry() const {
    return geometry_;
  }

  /// Helper function to normalize the recorded flux in a post-processing step.
  ///
  /// SOURCE divides by the number of rays per unit source area and by the
  /// normalization area of the sphere, so the result is comparable between
  /// runs with a different ray count or a different geometry size.
  ///
  /// MAX only rescales to the largest recorded value. Since that reference
  /// moves as the surface evolves, it must not be used when the flux feeds
  /// back into an absolute rate.
  void
  normalizeFlux(std::vector<NumericType> &flux,
                NormalizationType norm = NormalizationType::SOURCE) override {
    assert(flux.size() == geometry_.getNumPrimitives() &&
           "Unequal number of points in normalizeFlux");

    switch (norm) {
    case NormalizationType::MAX: {
      const auto maxv = *std::max_element(flux.begin(), flux.end());
      if (maxv <= 0)
        break;
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] /= maxv;
      }
      break;
    }

    case NormalizationType::SOURCE: {
      if (!this->pSource_) {
        VIENNACORE_LOG_WARNING(
            "No source was specified in rayTrace for the normalization.");
        break;
      }
      NumericType sourceArea = this->pSource_->getSourceArea();
      auto numTotalRays =
          this->config_.numRaysFixed == 0
              ? this->pSource_->getNumPoints() * this->config_.numRaysPerPoint
              : this->config_.numRaysFixed;
      const NumericType normFactor = sourceArea / numTotalRays;
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        const NumericType area = geometry_.getNormalizationArea(idx);
        flux[idx] = area > 0 ? flux[idx] * normFactor / area : NumericType(0);
      }
      break;
    }

    default:
      break;
    }
  }

  /// Averaging the flux over the neighborhood would mix atoms that sit in
  /// different layers, which is exactly the information the sphere geometry is
  /// supposed to resolve. Smoothing is therefore not applied.
  void smoothFlux(std::vector<NumericType> &flux, int numNeighbors) override {}

private:
  void checkSettings() {
    this->RTInfo_.error = false;
    if (this->pParticle_ == nullptr) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No particle was specified in rayTrace. Aborting.");
    }
    if (geometry_.checkGeometryEmpty()) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No geometry was passed to rayTrace. Aborting.");
    }
    if (geometry_.getMaxRadius() <= 0) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("Sphere radius is zero in rayTrace. Aborting.");
    }
    if (warned_)
      return;
    warned_ = true;
    if (!geometry_.hasNormalizationArea()) {
      VIENNACORE_LOG_WARNING(
          "TraceSphere: no normalization area was set, falling back to "
          "2*pi*r^2. That is the capture cross section of an isolated sphere "
          "and underestimates the flux of a close packed surface.");
    }
    if (useLateralBox_)
      return;
    auto const traceSettings =
        rayInternal::getTraceSettings(this->sourceDirection_);
    constexpr int numLateralDirs = D == 3 ? 2 : 1;
    for (int i = 0; i < numLateralDirs; ++i) {
      if (this->boundaryConditions_[traceSettings[1 + i]] ==
          BoundaryCondition::PERIODIC_BOUNDARY) {
        VIENNACORE_LOG_WARNING(
            "TraceSphere: periodic boundary without an explicit lateral "
            "bounding box. The period is derived from the sphere centers and "
            "is therefore one atomic spacing too short. Use "
            "setLateralBoundingBox().");
        break;
      }
    }
  }

private:
  GeometrySphere<NumericType, D> geometry_;

  Vec3D<NumericType> lateralMin_{0, 0, 0};
  Vec3D<NumericType> lateralMax_{0, 0, 0};
  bool useLateralBox_ = false;
  bool warned_ = false;
};

} // namespace viennaray
