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

#include <iostream>
#include <fstream>
#include <sstream>
#include "spatialDecomposition.hpp"
#include "json.hpp"

using json = nlohmann::json;

int main() {

  std::stringstream ss;

  std::vector<float> mic_locs(6*3, 0.f);

  mic_locs.at(0) = 0.025f;
  mic_locs.at(3) = -0.025f;
  mic_locs.at(7) = 0.025f;
  mic_locs.at(10) = -0.025f;
  mic_locs.at(14) = 0.025f;
  mic_locs.at(17) = -0.025f;

  SpatialDecomposer sd;

  ss.str("");
  ss<<"audio/livingroom.json";
  std::cout<<ss.str()<<std::endl;
  std::ifstream file(ss.str());
  json j;
  j<<file;
  file.close();

  std::vector< float* > irs;
  std::vector< float > ir0 = j["ir0"].get< std::vector<float> >();
  std::vector< float > ir1 = j["ir1"].get< std::vector<float> >();
  std::vector< float > ir2 = j["ir2"].get< std::vector<float> >();
  std::vector< float > ir3 = j["ir3"].get< std::vector<float> >();
  std::vector< float > ir4 = j["ir4"].get< std::vector<float> >();
  std::vector< float > ir5 = j["ir5"].get< std::vector<float> >();

  irs.push_back(&(ir0[0]));
  irs.push_back(&(ir1[0]));
  irs.push_back(&(ir2[0]));
  irs.push_back(&(ir3[0]));
  irs.push_back(&(ir4[0]));
  irs.push_back(&(ir5[0]));

  unsigned int fs = j["fs"].get<unsigned int>();
  unsigned int filter_len = j["filter_len"].get< unsigned int >();
  unsigned int frame_len = 36;

  sd.initialize(fs, frame_len, filter_len, &mic_locs[0], mic_locs.size()/3);

  sd.processIRs(&(irs[0]));

  ss.str("");
  ss<<"audio/living_room_locs.json";
  std::ofstream fileo(ss.str(), std::ofstream::out);
  json j2;
  j2["az"] = sd.ret_az_;
  j2["el"] = sd.ret_el_;
  j2["p"] = ir0;
  fileo<<j2;
  fileo.close();

  sd.destroy();
}
