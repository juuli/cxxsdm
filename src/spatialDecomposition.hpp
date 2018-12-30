#ifndef SPATIAL_DECOMPOSITION_HPP
#define SPATIAL_DECOMPOSITION_HPP

////////////////////////////////////////////////////////////////////////////////
//
// This file is a part of the CxxSDM spatial decomposition
// library. It is released under the MIT License. You should have
// received a copy of the MIT License along with CxxSDM.  If not, see
// http://www.opensource.org/licenses/mit-license.php
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// For details, see the LICENSE file
//
// (C) 2018 Jukka Saarelma
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#ifdef _WIN32
  #define DLLEXPORT extern "C" __declspec( dllexport )
#else
  #define DLLEXPORT
#endif

#define EIGEN_USE_LAPACKE

// C interface
extern "C" {
  DLLEXPORT void* initialize(unsigned int fs, unsigned int frame_len,
                   unsigned int resp_len, float* mic_locs,
                   unsigned int num_mics);

  DLLEXPORT void destroy(void* sd_);

  DLLEXPORT void processIRs(void* sd_, float* irs, float* az_out, float* el_out);

  DLLEXPORT void synthFromLocs(float* p, float* az0, float* el0, float* az1, float* el1,
                     unsigned int resp_len, unsigned int ls_num, float* ret);
}

inline Eigen::Vector3f cart2sph(Eigen::Vector3f cart) {
  float hypotXY = hypot(cart.x(), cart.y());
  float r = hypot(hypotXY, cart.z());
  float elev = atan2(cart.z(), hypotXY)/M_PI*180.f;
  float az = atan2(cart.y(), cart.x())/M_PI*180.f;
  return Eigen::Vector3f(az, elev, r);
}

inline Eigen::Vector3f sph2cart(Eigen::Vector3f sph) {
  float z = sph.z()*sinf(sph.y()/180*M_PI);
  float el_cos = sph.z()*cosf(sph.y()/180*M_PI);
  float x = el_cos*cosf(sph.x()/180*M_PI);
  float y = el_cos*sinf(sph.x()/180*M_PI);
  return Eigen::Vector3f(x,y,z);
}

// From http://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
template<typename _Matrix_Type_>
inline bool pinv(const _Matrix_Type_ &a, _Matrix_Type_ &result,
                 double epsilon = std::numeric_limits<double>::epsilon()) {
  if(a.rows() < a.cols()) {
    printf("Inverse false rows: %ld cols: %ld \n", a.rows(), a.cols());
    return false;
  }
  Eigen::JacobiSVD< _Matrix_Type_ > svd = a.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon*std::max(a.cols(), a.rows())*
                     svd.singularValues().array().abs().maxCoeff();

  result = svd.matrixV()*
           ((svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0) ).matrix().asDiagonal()*
           svd.matrixU().adjoint();
  return true;
}

class SpatialDecomposer {
public:
  SpatialDecomposer() {};
  ~SpatialDecomposer() {};

  void initialize(unsigned int fs, unsigned int winlen, unsigned int resp_len,
                  float* mic_locs, unsigned int num_mics);

  void correlate(float* a, float* b, float* out);

  void destroy();

  void processFrame(float* frame);

  void processIRs(float** irs);

private:

  std::vector<unsigned int> comb(int N, int K) {
    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    std::vector<unsigned int> ret;
    // print integers and permute bitmask
    do {
      for (int i = 0; i < N; ++i) { // [0..N-1] integers
          if (bitmask[i]) ret.push_back(i);
      }
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return ret;
  }

  std::vector<float> hanning(unsigned int len) {
    std::vector<float> ret(len);
    for(unsigned int i = 0; i < len; i++) {
        float multiplier = 0.5f * (1.f - cosf(2.0*M_PI*(float)i/(float)len));
        ret.at(i) = multiplier;
    }
    return ret;
  }

  unsigned int maxIdx(float* data, unsigned int offset, unsigned int max_dist_i);

  //////////////////////////////////////////////////////////////////////////////
  // Interpolate by fitting a gaussian function f = a*exp(-b(x-c)**2) to three
  // values around the maximum value of the cross-correlation vector.
  //
  // Zhang, Lei, and Xiaolin Wu. "On cross correlation based-discrete time delay
  // estimation." Acoustics, Speech, and Signal Processing, 2005. Proceedings.
  // (ICASSP'05). IEEE International Conference on. Vol. 4. IEEE, 2005.
  //////////////////////////////////////////////////////////////////////////////

  float interpolateTau(float* data, unsigned int i);

// Keeping the fields public for now for debugging
public:
  fftwf_plan fft_;
  fftwf_plan ifft_;
  std::vector< fftwf_complex* > f_buffers_;
  std::vector< float* > t_buffers_;
  std::vector< float > han_window_;

  std::vector< fftwf_complex* > corr_buffers_;
  std::vector< float* > real_corr_buffers_;

  std::vector<float> ret_p_;
  std::vector<float> ret_az_;
  std::vector<float> ret_el_;

  unsigned int fs_;
  unsigned int frame_len_;
  unsigned int complex_len_;
  unsigned int transform_len_;
  unsigned int transform_len_dif_;
  unsigned int num_win_;
  unsigned int resp_len_;
  unsigned int num_mics_;
  unsigned int num_mic_pairs_;
  unsigned int max_dist_i_;
  unsigned int offset_;
  Eigen::MatrixXf mic_locs_;
  Eigen::MatrixXf ls_locs_;
  Eigen::MatrixXi mic_pairs_;
  Eigen::MatrixXf V_;
  Eigen::MatrixXf inv_V_;
  Eigen::MatrixXf pair_dists_;
  Eigen::MatrixXf tau_;
  float c_;

};

#endif
