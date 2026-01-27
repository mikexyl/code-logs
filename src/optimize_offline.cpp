#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace std;
using namespace gtsam;

// Structure to hold robot information
struct RobotGraph {
  int robot_id;
  string filename;
  NonlinearFactorGraph::shared_ptr graph;
  Values::shared_ptr initial;
};

void writeG2oNoIndex(const NonlinearFactorGraph &graph, const Values &estimate,
                     const string &filename) {
  fstream stream(filename.c_str(), fstream::out);

  // Use a lambda here to more easily modify behavior in future.
  auto index = [](gtsam::Key key) { return key; };

  // save 2D poses
  for (const auto &pair : estimate.extract<Pose2>()) {
    const Pose2 &pose = pair.second;
    stream << "VERTEX_SE2 " << index(pair.first) << " " << pose.x() << " "
           << pose.y() << " " << pose.theta() << endl;
  }

  // save 3D poses
  for (const auto &pair : estimate.extract<Pose3>()) {
    const Pose3 &pose = pair.second;
    const Point3 t = pose.translation();
    const auto q = pose.rotation().toQuaternion();
    stream << "VERTEX_SE3:QUAT " << index(pair.first) << " " << t.x() << " "
           << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " "
           << q.z() << " " << q.w() << endl;
  }

  // save 2D landmarks
  for (const auto &pair : estimate.extract<Point2>()) {
    const Point2 &point = pair.second;
    stream << "VERTEX_XY " << index(pair.first) << " " << point.x() << " "
           << point.y() << endl;
  }

  // save 3D landmarks
  for (const auto &pair : estimate.extract<Point3>()) {
    const Point3 &point = pair.second;
    stream << "VERTEX_TRACKXYZ " << index(pair.first) << " " << point.x() << " "
           << point.y() << " " << point.z() << endl;
  }

  // save edges (2D or 3D)
  for (const auto &factor_ : graph) {
    auto factor = boost::dynamic_pointer_cast<BetweenFactor<Pose2>>(factor_);
    if (factor) {
      SharedNoiseModel model = factor->noiseModel();
      auto gaussianModel =
          boost::dynamic_pointer_cast<noiseModel::Gaussian>(model);
      if (!gaussianModel) {
        model->print("model\n");
        throw invalid_argument("writeG2o: invalid noise model!");
      }
      Matrix3 Info = gaussianModel->R().transpose() * gaussianModel->R();
      Pose2 pose = factor->measured(); //.inverse();
      stream << "EDGE_SE2 " << index(factor->key<1>()) << " "
             << index(factor->key<2>()) << " " << pose.x() << " " << pose.y()
             << " " << pose.theta();
      for (size_t i = 0; i < 3; i++) {
        for (size_t j = i; j < 3; j++) {
          stream << " " << Info(i, j);
        }
      }
      stream << endl;
    }

    auto factor3D = boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(factor_);

    if (factor3D) {
      SharedNoiseModel model = factor3D->noiseModel();

      boost::shared_ptr<noiseModel::Gaussian> gaussianModel =
          boost::dynamic_pointer_cast<noiseModel::Gaussian>(model);
      if (!gaussianModel) {
        model->print("model\n");
        throw invalid_argument("writeG2o: invalid noise model!");
      }
      Matrix6 Info = gaussianModel->R().transpose() * gaussianModel->R();
      const Pose3 pose3D = factor3D->measured();
      const Point3 p = pose3D.translation();
      const auto q = pose3D.rotation().toQuaternion();
      stream << "EDGE_SE3:QUAT " << index(factor3D->key<1>()) << " "
             << index(factor3D->key<2>()) << " " << p.x() << " " << p.y() << " "
             << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
             << q.w();

      // g2o's EDGE_SE3:QUAT stores information/precision of Pose3 in t,R order,
      // unlike GTSAM:
      Matrix6 InfoG2o = I_6x6;
      InfoG2o.block<3, 3>(0, 0) = Info.block<3, 3>(3, 3); // cov translation
      InfoG2o.block<3, 3>(3, 3) = Info.block<3, 3>(0, 0); // cov rotation
      InfoG2o.block<3, 3>(0, 3) =
          Info.block<3, 3>(3, 0); // off diagonal R,t -> t,R
      InfoG2o.block<3, 3>(3, 0) =
          Info.block<3, 3>(0, 3); // off diagonal t,R -> R,t

      for (size_t i = 0; i < 6; i++) {
        for (size_t j = i; j < 6; j++) {
          stream << " " << InfoG2o(i, j);
        }
      }
      stream << endl;
    }
  }
  stream.close();
}

/**
 * Find all g2o files matching the pattern in subdirectories
 */
vector<string> findG2oFiles(const string &base_path) {
  vector<string> files;

  // Expected pattern: base_path/*/dpgo/bpsam_robot_*.g2o
  if (fs::exists(base_path) && fs::is_directory(base_path)) {
    for (const auto &entry : fs::directory_iterator(base_path)) {
      if (entry.is_directory()) {
        fs::path subdir = entry.path() / "dpgo";
        if (fs::exists(subdir) && fs::is_directory(subdir)) {
          for (const auto &sub_entry : fs::directory_iterator(subdir)) {
            if (sub_entry.is_regular_file()) {
              string filename = sub_entry.path().filename().string();
              if (filename.find("bpsam_robot_") == 0 &&
                  sub_entry.path().extension() == ".g2o") {
                files.push_back(sub_entry.path().string());
                cout << "Found: " << sub_entry.path().string() << endl;
              }
            }
          }
        }
      }
    }
  }

  sort(files.begin(), files.end());
  return files;
}

/**
 * Load a single g2o file
 */
RobotGraph loadG2oFile(const string &filename, int robot_id) {
  RobotGraph robot_graph;
  robot_graph.robot_id = robot_id;
  robot_graph.filename = filename;

  cout << "\nLoading " << filename << " as robot " << robot_id << "..." << endl;

  try {
    tie(robot_graph.graph, robot_graph.initial) = readG2o(filename, true);
    cout << "Successfully loaded: " << robot_graph.graph->size() << " factors, "
         << robot_graph.initial->size() << " poses" << endl;
  } catch (const gtsam::ValuesKeyAlreadyExists &e) {
    cerr << "Error loading " << filename << ": "
         << MultiRobotKeyFormatter(e.key()) << " already exists." << endl;
    robot_graph.graph = boost::make_shared<NonlinearFactorGraph>();
    robot_graph.initial = boost::make_shared<Values>();
  }

  return robot_graph;
}

/**
 * Combine multiple robot graphs into a single graph
 */
pair<NonlinearFactorGraph, Values>
combineGraphs(const vector<RobotGraph> &robot_graphs) {
  NonlinearFactorGraph combined_graph;
  Values combined_initial;

  for (const auto &robot : robot_graphs) {
    // Add all factors from this robot
    for (const auto &factor : *robot.graph) {
      if (factor) {
        combined_graph.add(factor);
      }
    }

    // Add all initial values from this robot
    combined_initial.insert_or_assign(*robot.initial);
  }

  return make_pair(combined_graph, combined_initial);
}

/**
 * Optimize the pose graph using GTSAM
 */
Values optimizeGraph(const NonlinearFactorGraph &graph, const Values &initial) {
  cout << "\n=== Optimizing graph ===" << endl;
  cout << "Initial error: " << graph.error(initial) << endl;

  // check if the graph has only one prior
  int n_priors = 0;
  for (const auto &factor : graph) {
    if (boost::dynamic_pointer_cast<PriorFactor<Pose2>>(factor) ||
        boost::dynamic_pointer_cast<PriorFactor<Pose3>>(factor)) {
      n_priors++;
    }
  }
  if (n_priors > 1) {
    cerr << "Error: Graph has more than one prior factor (" << n_priors
         << "). Optimization aborted." << endl;
    return initial;
  }

  // set all odometry factors as known inliers
  GncParams<GaussNewtonParams>::IndexVector known_inliers;
  for (size_t i = 0; i < graph.size(); i++) {
    auto factor = graph.at(i);
    if (factor) {
      auto between3D =
          boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(factor);
      // check key index if it is odometry
      if (between3D) {
        Key r1 = Symbol(between3D->key<1>()).chr();
        Key r2 = Symbol(between3D->key<2>()).chr();
        if (r1 == r2) {
          known_inliers.push_back(i);
        }
      }
    }
  }

  std::cout << "Known inliers count: " << known_inliers.size() << std::endl;
  std::cout << "total factors: " << graph.size() << std::endl;

  // Optimize
  // GncParams<GaussNewtonParams> params;
  // params.setLossType(GncLossType::GM);
  // params.setKnownInliers(known_inliers);
  // params.setVerbosityGNC(GncParams<GaussNewtonParams>::SUMMARY);
  // GncOptimizer<GncParams<GaussNewtonParams>> optimizer(graph, initial,
  // params); Values result = optimizer.optimize();

  LevenbergMarquardtParams params;
  params.setVerbosity("SUMMARY");
  LevenbergMarquardtOptimizer optimizer(graph, initial, params);
  Values result = optimizer.optimize();

  cout << "Final error: " << graph.error(result) << endl;
  cout << "Optimization complete!" << endl;

  return result;
}

/**
 * Save optimized results to file
 */
void saveResults(const string &output_dir, const NonlinearFactorGraph &graph,
                 const Values &initial, const Values &optimized,
                 int num_robots) {
  cout << "\n=== Saving results ===" << endl;

  // Create output directory if it doesn't exist
  fs::create_directories(output_dir);

  // Save combined g2o file
  string combined_file = output_dir + "/combined_optimized.g2o";
  writeG2oNoIndex(graph, optimized, combined_file);
  cout << "Saved combined graph to: " << combined_file << endl;

  // Save per-robot trajectories
  for (int robot_id = 0; robot_id < num_robots; robot_id++) {
    int key_offset = robot_id * 10000;

    string robot_initial_file =
        output_dir + "/robot_" + to_string(robot_id) + "_initial.txt";
    string robot_optimized_file =
        output_dir + "/robot_" + to_string(robot_id) + "_optimized.txt";

    ofstream initial_stream(robot_initial_file);
    ofstream optimized_stream(robot_optimized_file);

    // Write header
    initial_stream << "# robot_id pose_id x y theta" << endl;
    optimized_stream
        << "# robot_id pose_id x y theta (or x y z qx qy qz qw for 3D)" << endl;

    // Extract poses for this robot
    for (const auto &key_value : initial) {
      Key key = key_value.key;
      if (key >= key_offset && key < key_offset + 10000) {
        int pose_id = key - key_offset;

        if (initial.exists<Pose2>(key)) {
          Pose2 pose_init = initial.at<Pose2>(key);
          Pose2 pose_opt = optimized.at<Pose2>(key);

          initial_stream << robot_id << " " << pose_id << " " << pose_init.x()
                         << " " << pose_init.y() << " " << pose_init.theta()
                         << endl;

          optimized_stream << robot_id << " " << pose_id << " " << pose_opt.x()
                           << " " << pose_opt.y() << " " << pose_opt.theta()
                           << endl;
        } else if (initial.exists<Pose3>(key)) {
          Pose3 pose_init = initial.at<Pose3>(key);
          Pose3 pose_opt = optimized.at<Pose3>(key);

          auto t_init = pose_init.translation();
          auto q_init = pose_init.rotation().toQuaternion();
          auto t_opt = pose_opt.translation();
          auto q_opt = pose_opt.rotation().toQuaternion();

          initial_stream << robot_id << " " << pose_id << " " << t_init.x()
                         << " " << t_init.y() << " " << t_init.z() << " "
                         << q_init.x() << " " << q_init.y() << " " << q_init.z()
                         << " " << q_init.w() << endl;

          optimized_stream << robot_id << " " << pose_id << " " << t_opt.x()
                           << " " << t_opt.y() << " " << t_opt.z() << " "
                           << q_opt.x() << " " << q_opt.y() << " " << q_opt.z()
                           << " " << q_opt.w() << endl;
        }
      }
    }

    initial_stream.close();
    optimized_stream.close();

    cout << "Saved robot " << robot_id << " trajectories" << endl;
  }

  // Save summary statistics
  string summary_file = output_dir + "/summary.txt";
  ofstream summary(summary_file);
  summary << "=== Optimization Summary ===" << endl;
  summary << "Number of robots: " << num_robots << endl;
  summary << "Total poses: " << initial.size() << endl;
  summary << "Total factors: " << graph.size() << endl;
  summary << "Initial error: " << graph.error(initial) << endl;
  summary << "Final error: " << graph.error(optimized) << endl;
  summary << "Error reduction: "
          << (1.0 - graph.error(optimized) / graph.error(initial)) * 100.0
          << "%" << endl;
  summary.close();

  cout << "Saved summary to: " << summary_file << endl;
}

/**
 * Main function
 */
int main(int argc, char **argv) {
  cout << "=== Multi-Robot Pose Graph Optimization ===" << endl;

  // Configuration

  string dataset = "gate";
  if (argc > 1) {
    dataset = argv[1];
  }

  string base_path = "/workspaces/src/code-logs/" + dataset;
  string output_dir = base_path + "/optimized_results";

  // Allow output directory override
  if (argc > 2) {
    output_dir = argv[2];
  }

  cout << "Base path: " << base_path << endl;
  cout << "Output directory: " << output_dir << endl;

  try {
    // Step 1: Find all g2o files
    cout << "\n=== Step 1: Finding g2o files ===" << endl;
    vector<string> g2o_files = findG2oFiles(base_path);

    if (g2o_files.empty()) {
      cerr << "Error: No g2o files found in " << base_path << endl;
      return 1;
    }

    cout << "Found " << g2o_files.size() << " g2o files" << endl;

    // Step 2: Load all g2o files
    cout << "\n=== Step 2: Loading g2o files ===" << endl;
    vector<RobotGraph> robot_graphs;
    for (size_t i = 0; i < g2o_files.size(); i++) {
      robot_graphs.push_back(loadG2oFile(g2o_files[i], i));
      std::cout << "Robot " << i << ": " << robot_graphs.back().graph->size()
                << " factors, " << robot_graphs.back().initial->size()
                << " poses" << std::endl;
    }

    // Step 3: Combine graphs
    auto [combined_graph, combined_initial] = combineGraphs(robot_graphs);
    cout << "\n=== Step 3: Combined graph ===" << endl;
    cout << "Total factors: " << combined_graph.size() << endl;
    cout << "Total poses: " << combined_initial.size() << endl;

    // add a prior at the first pose to fix the gauge freedom
    auto first_key = combined_graph.front()->keys().front();
    combined_graph.addPrior(
        first_key, Pose3::Identity(),
        noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(1e-6), Vector3::Constant(1e-6))
                .finished()));

    // Step 5: Optimize
    Values optimized = optimizeGraph(combined_graph, combined_initial);

    // Step 6: Save results
    saveResults(output_dir, combined_graph, combined_initial, optimized,
                robot_graphs.size());

    cout << "\n=== SUCCESS ===" << endl;
    cout << "Results saved to: " << output_dir << endl;

  } catch (const exception &e) {
    cerr << "\nError: " << e.what() << endl;
    return 1;
  }

  return 0;
}
