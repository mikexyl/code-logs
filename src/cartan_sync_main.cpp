/**
 * Cartan-Sync wrapper: reads a g2o file, runs CartanSync, and writes the
 * solved poses back to a g2o file in VERTEX_SE3:QUAT format.
 *
 * Usage:
 *   cartan_sync <input.g2o> [output.g2o]
 *
 * If output path is omitted, writes to <input>.result.g2o
 * Progress / status is printed to stderr; the output file contains only
 * VERTEX_SE3:QUAT lines.
 *
 * NOTE on vertex ID convention:
 *   read_g2o_file() uses the raw vertex IDs from edge endpoints as indices
 *   directly.  Our merged g2o uses 0-based sequential IDs 0..n-1, so result
 *   column k (Rhat block k, that col k) corresponds to vertex ID k.
 *   ID 0 is SE-Sync's anchor pose (translation = 0).
 */

#include "SESync.h"
#include "SESync_utils.h"

#include <Eigen/Geometry>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace SESync;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input.g2o> [output.g2o]" << endl;
        return 1;
    }

    string input_path  = argv[1];
    string output_path = (argc >= 3)
        ? argv[2]
        : (input_path + ".result.g2o");

    // -----------------------------------------------------------------------
    // Load
    // -----------------------------------------------------------------------
    size_t num_poses = 0;
    vector<RelativePoseMeasurement> measurements =
        read_g2o_file(input_path, num_poses);
    cerr << "Loaded " << measurements.size() << " measurements between "
         << num_poses << " poses from " << input_path << endl;

    // -----------------------------------------------------------------------
    // Solve
    // -----------------------------------------------------------------------
    SESyncOpts opts;
    opts.verbose      = false;
    opts.eig_comp_tol = 1e-6;
    opts.min_eig_num_tol = 1e-3;

    SESyncResult results = SESync::SESync(measurements, AlgType::CartanSync, opts);

    // Status report
    string status_str;
    switch (results.status) {
        case GLOBAL_OPT:      status_str = "GLOBAL_OPT";      break;
        case SADDLE_POINT:    status_str = "SADDLE_POINT";    break;
        case EIG_IMPRECISION: status_str = "EIG_IMPRECISION"; break;
        case RS_ITER_LIMIT:   status_str = "RS_ITER_LIMIT";   break;
        default:              status_str = "UNKNOWN";         break;
    }
    cerr << "Status: " << status_str
         << "  Fxhat=" << results.Fxhat
         << "  gradnorm=" << results.gradnorm << endl;

    // -----------------------------------------------------------------------
    // Write poses: VERTEX_SE3:QUAT id x y z qx qy qz qw
    // -----------------------------------------------------------------------
    ofstream out(output_path);
    if (!out) {
        cerr << "Failed to open output file: " << output_path << endl;
        return 1;
    }
    out << fixed << setprecision(10);

    // Sanity check result dimensions
    const size_t n_r = results.Rhat.cols() / 3;
    const size_t n_t = results.that.cols();
    cerr << "Rhat: " << results.Rhat.rows() << "x" << results.Rhat.cols()
         << "  that: " << results.that.rows() << "x" << results.that.cols()
         << "  num_poses: " << num_poses << endl;
    if (n_r == 0 || n_t == 0) {
        cerr << "Error: result matrices are empty (optimization likely failed)" << endl;
        return 1;
    }
    const size_t n_out = std::min({num_poses, n_r, n_t});

    // Iterate k = 0 .. n_out-1 (all poses; k=0 is the anchor at origin)
    for (size_t k = 0; k < n_out; ++k) {
        Eigen::Matrix3d    R = results.Rhat.block<3, 3>(0, 3 * k);
        Eigen::Vector3d    t = results.that.col(k);
        Eigen::Quaterniond q(R);
        q.normalize();

        out << "VERTEX_SE3:QUAT " << k
            << " " << t(0) << " " << t(1) << " " << t(2)
            << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << "\n";
    }
    out.close();
    cerr << "Wrote " << n_out << " poses to " << output_path << endl;

    // Exit immediately to avoid ROPTLIB destructor double-free (known upstream bug).
    std::exit(0);
}
