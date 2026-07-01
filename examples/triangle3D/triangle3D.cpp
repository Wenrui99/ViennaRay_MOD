#include <rayTraceTriangle.hpp>
#include <rayUtil.hpp>

#include <vcTimer.hpp>

#include <omp.h>

#include <vector>

using namespace viennaray;

int main() {
  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<unsigned>> triangles;
  NumericType gridDelta;
  rayInternal::readMeshFromFile<NumericType, D>("trenchMesh.dat", gridDelta,
                                                points, triangles);

  TriangleMesh mesh(points, triangles, gridDelta);
  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(mesh);

  // ==========================================================================
  // Mode 1: global sticking probability (original behaviour).
  // Every triangle uses the same sticking probability.
  // --------------------------------------------------------------------------
  // NumericType stickingProbability = 0.1;
  // auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
  //     stickingProbability, "flux");
  // tracer.setParticleType(particle);
  // // ==========================================================================

  // ==========================================================================
  // Mode 2: per-triangle sticking probability.
  // The sticking probability is looked up by triangle ID (primID).
  // --------------------------------------------------------------------------
  NumericType defaultStickingProbability = 0.1;
  // stickingProbabilities[i] is the sticking probability of triangle i.
  std::vector<NumericType> stickingProbabilities(triangles.size(),
                                                 defaultStickingProbability);
  // --- placeholder values for testing ---
  for (size_t i = 0; i < triangles.size(); ++i) {
    // e.g. give the first half and second half different sticking values.
    stickingProbabilities[i] = (i < triangles.size() / 2) ? 0.05 : 0.5;
  }
  // --------------------------------------
  auto particle =
      std::make_unique<DiffuseParticleMultiSticking<NumericType, D>>(
          stickingProbabilities, "flux", defaultStickingProbability);
  tracer.setParticleType(particle);
  // ==========================================================================

  tracer.setNumberOfRaysPerPoint(2000);

  Timer timer;
  timer.start();
  tracer.apply();
  timer.finish();

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " s\n";

  auto flux = *tracer.getLocalData().getScalarData("flux");
  tracer.normalizeFlux(flux, NormalizationType::SOURCE);

  rayInternal::writeVTP<NumericType, D>("triangleGeometryOutput_test.vtp", points,
                                        triangles, flux);
}
