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

#include <float.h>
#include "spatialDecomposition.hpp"

#define PROGRESS_MOD 10000

extern "C" {
  DLLEXPORT void* initialize(unsigned int fs, unsigned int frame_len,
                             unsigned int resp_len, float* mic_locs,
                             unsigned int num_mics) {
    SpatialDecomposer* sd = new SpatialDecomposer();
    sd->initialize(fs, frame_len, resp_len, mic_locs, num_mics);
    return (void*)sd;
  }

  DLLEXPORT void destroy(void* sd_) {
    SpatialDecomposer* sd = (SpatialDecomposer*)sd_;
    sd->destroy();
    delete sd;
  }

  DLLEXPORT void processIRs(void* sd_, float* irs, float* az_out, float* el_out) {
    SpatialDecomposer* sd = (SpatialDecomposer*)sd_;
    std::vector< float* > irs_v(sd->num_mics_);
    for(unsigned int i = 0; i < sd->num_mics_; i++)
      irs_v.at(i) = irs+i*sd->resp_len_;
    sd->processIRs(&irs_v[0]);
    memcpy(az_out, &sd->ret_az_[0], sd->resp_len_*sizeof(float));
    memcpy(el_out, &sd->ret_el_[0], sd->resp_len_*sizeof(float));
  }

  DLLEXPORT float centralAngle(float az0, float el0, float az1, float el1) {
    return acosf(sinf(el0)*sinf(el1)+cosf(el0)*cosf(el1)*cosf(abs(az1-az0)));
  }

  DLLEXPORT void synthFromLocs(float* p, float* az0, float* el0, float* az1, float* el1,
                               unsigned int resp_len, unsigned int num_ls, float* ret) {

    for(unsigned int i = 0; i < resp_len; i++) {
      if(i%PROGRESS_MOD==0)
        std::cout<<"synthFromLoc: "<<i<<" / "<<resp_len<<std::endl;

      float min_coef = 1e9f;
      unsigned int min_i = 0;
      for(unsigned int j = 0; j < num_ls; j++){
        float cur = centralAngle(az0[i], el0[i], az1[j], el1[j]);
        if(cur<min_coef) {
          min_coef = cur;
          min_i = j;
        }
      }
      // Direct panning
      ret[resp_len*min_i+i] = p[i];
    }
    return;
  }
}

unsigned int nextPow2(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void SpatialDecomposer::initialize(unsigned int fs, unsigned int frame_len,
                                  unsigned int resp_len, float* mic_locs,
                                  unsigned int num_mics) {
  this->c_ = 344.f;
  this->fs_ = fs;
  this->frame_len_= frame_len;
  this->transform_len_ = nextPow2(this->frame_len_);
  this->transform_len_dif_ = this->transform_len_-this->frame_len_;
  this->complex_len_ = this->transform_len_/2+1;
  this->offset_ = this->transform_len_/2;
  this->resp_len_ = resp_len;
  this->num_mics_ = num_mics;
  this->mic_locs_ = Eigen::MatrixXf(this->num_mics_, 3);

  for(unsigned int i = 0; i < this->num_mics_; i++) {
    for(unsigned int j = 0; j < 3; j++)
      this->mic_locs_(i,j) = mic_locs[i*3+j];
  }

  std::vector<unsigned int> combinations = this->comb(this->num_mics_, 2);
  this->num_mic_pairs_ = combinations.size()/2;
  this->mic_pairs_ = Eigen::MatrixXi(this->num_mic_pairs_,2);

  for(unsigned int i = 0; i < this->num_mic_pairs_; i++) {
    for(unsigned int j = 0; j < 2; j++)
      this->mic_pairs_(i, j) = (float)combinations.at(j+i*2);
  }

  this->V_ = Eigen::MatrixXf(this->num_mic_pairs_, 3);
  this->pair_dists_ = Eigen::MatrixXf(this->num_mic_pairs_, 1);
  for(unsigned int i = 0; i < this->num_mic_pairs_; i++) {
    Eigen::Vector3f r1 = this->mic_locs_.row(this->mic_pairs_(i,0));
    Eigen::Vector3f r2 = this->mic_locs_.row(this->mic_pairs_(i,1));
    this->V_.row(i)=r1-r2;
    this->pair_dists_(i,0) = (r1-r2).norm();
  }

  pinv(this->V_, this->inv_V_);
  this->inv_V_ *= -1.f;

  this->max_dist_i_ = ceil(this->pair_dists_.maxCoeff()/this->c_*(float)this->fs_);
  this->tau_ = Eigen::MatrixXf(this->num_mic_pairs_, 1);
  this->num_win_ = this->resp_len_/this->frame_len_;

  for(unsigned int i = 0; i < this->num_mics_; i++) {
    this->f_buffers_.push_back((fftwf_complex*)calloc(this->complex_len_, sizeof(fftwf_complex)));
    this->t_buffers_.push_back((float*)calloc(this->transform_len_, sizeof(float)));
  }

  this->han_window_ = this->hanning(this->frame_len_);

  for(unsigned int i = 0; i < this->num_mic_pairs_; i++) {
    this->corr_buffers_.push_back((fftwf_complex*)calloc(this->complex_len_,
                                                sizeof(fftwf_complex)));

    this->real_corr_buffers_.push_back((float*)calloc(this->transform_len_,
                                             sizeof(float)));
  }

  this->fft_ = fftwf_plan_dft_r2c_1d(this->transform_len_,
                                     this->t_buffers_.at(0),
                                     this->f_buffers_.at(0),
                                     FFTW_MEASURE);

  this->ifft_ = fftwf_plan_dft_c2r_1d(this->transform_len_,
                                      this->f_buffers_.at(0),
                                      this->t_buffers_.at(0),
                                      FFTW_MEASURE);

  this->ret_p_.assign(resp_len, 0.f);
  this->ret_az_.assign(resp_len, 0.f);
  this->ret_el_.assign(resp_len, 0.f);

  std::cout<<"------------------"<<std::endl;
  std::cout<<"Init"<<std::endl;
  std::cout<<"Resp len: "<<this->resp_len_<<std::endl;
  std::cout<<"Frame len: "<<this->frame_len_<<std::endl;
  std::cout<<"Num mics: "<<this->num_mics_<<std::endl;
  std::cout<<"------------------"<<std::endl;
}

void SpatialDecomposer::destroy() {
  for(unsigned int i = 0; i < this->num_mics_; i++) {
    free(this->f_buffers_.at(i));
    free(this->t_buffers_.at(i));
  }

  for(unsigned int i = 0; i < this->num_mic_pairs_; i++) {
    free(this->corr_buffers_.at(i));
    free(this->real_corr_buffers_.at(i));
  }

  fftwf_destroy_plan(this->fft_);
  fftwf_destroy_plan(this->ifft_);
  fftwf_cleanup();
}


void multiplyConjugate(fftwf_complex* a, fftwf_complex* b, fftwf_complex* output) {
  output[0][0] = a[0][0]*b[0][0]+a[0][1]*b[0][1];
  output[0][1] = a[0][0]*b[0][1]-a[0][1]*b[0][0];
  output[0][1] = output[0][1]*-1.f;
}

void blockShift(float* data, float* temp, unsigned int len) {
  unsigned int block_size = len/2;
  memcpy(temp, data, block_size*sizeof(float));
  memmove(data, data+block_size,  block_size*sizeof(float));
  memcpy(data+block_size, temp,  block_size*sizeof(float));
}


unsigned int SpatialDecomposer::maxIdx(float* data, unsigned int offset,
                                       unsigned int max_dist_i) {
  float mx = -1e9;
  unsigned int mx_i = offset;
  for(unsigned int i = offset-max_dist_i; i < offset+max_dist_i; i++) {
    if(data[i]>mx) {
      mx = data[i];
      mx_i = i;
    }
  }
  return mx_i;
}

float SpatialDecomposer::interpolateTau(float* data, unsigned int i) {
  float log_p_1 = logf(data[i+1]);
  float log_m_1 = logf(data[i-1]);
  float c = (log_p_1-log_m_1)/
            (4.f*logf(data[i])-2.0f*(log_m_1 - log_p_1));
  return  c + (float)i;
}

void SpatialDecomposer::processIRs(float** irs) {
  int num_frames = this->resp_len_/this->frame_len_;
  float* shift_block = (float*)calloc(this->transform_len_/2, sizeof(float));
  float inv_fs = 1.f/(float)this->fs_;

  for(unsigned int i = this->frame_len_/2; i < num_frames*this->frame_len_; i++) {
    if((i-this->frame_len_/2)%PROGRESS_MOD == 0)
      std::cout<<"Processing idx: "<<i-this->frame_len_/2<<std::endl;

    for(unsigned int j = 0; j < this->num_mics_; j++) {
      memcpy(this->t_buffers_[j],
             &irs[j][i-this->frame_len_/2], this->frame_len_*sizeof(float));

      for(unsigned int k = 0; k < this->frame_len_; k++)
        this->t_buffers_[j][k]*=this->han_window_.at(k);

      fftwf_execute_dft_r2c(this->fft_,
                            (this->t_buffers_[j]),
                            (this->f_buffers_[j]));
    }

    for(unsigned int j = 0; j < this->num_mic_pairs_; j++) {
      fftwf_complex* a = this->f_buffers_[this->mic_pairs_(j,0)];
      fftwf_complex* b = this->f_buffers_[this->mic_pairs_(j,1)];
      fftwf_complex* o = this->corr_buffers_[j];

      for(unsigned int k = 0; k < this->complex_len_; k++) {
        multiplyConjugate(&a[k], &b[k], &o[k]);
      }

      fftwf_execute_dft_c2r(this->ifft_,
                            this->corr_buffers_[j],
                            this->real_corr_buffers_[j]);

      for(unsigned int k = 0; k < this->transform_len_; k++) {
        if(this->real_corr_buffers_[j][k]<FLT_MIN)
           this->real_corr_buffers_[j][k] = FLT_MIN;
      }

      blockShift(this->real_corr_buffers_[j], shift_block, this->transform_len_);

      unsigned int mx_i = this->maxIdx(this->real_corr_buffers_[j],
                                       this->offset_,
                                       this->max_dist_i_);

      this->tau_(j, 0) = (this->interpolateTau(this->real_corr_buffers_[j], mx_i)-
                         (float)this->offset_)*inv_fs;

    }

    Eigen::Vector3f k = Eigen::Vector3f(this->inv_V_*this->tau_);
    Eigen::Vector3f sph = cart2sph(k);

    this->ret_az_.at(i) = sph.x();
    this->ret_el_.at(i) = sph.y();
  }

  free(shift_block);
  return;
}
