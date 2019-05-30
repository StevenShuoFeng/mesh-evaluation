#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cfloat>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// OpenMP
#include <omp.h>

#include "triangle_point/poitri.h"
#include "triangle_point/data_structures.h"


/** \brief Read all files in a directory matching the given extension.
 * \param[in] directory path to directory
 * \param[out] files read file paths
 * \param[in] extension extension to filter for
 */
void read_directory(const boost::filesystem::path directory, std::map<int, boost::filesystem::path>& files, const std::vector<std::string> &extensions) {

  files.clear();
  boost::filesystem::directory_iterator end;

  for (boost::filesystem::directory_iterator it(directory); it != end; ++it) {
    bool filtered = true;
    for (unsigned int i = 0; i < extensions.size(); i++) {
      if (it->path().extension().string() == extensions[i]) {
        filtered = false;
      }
    }

    if (!filtered) {
      int number = std::stoi(it->path().filename().string());
      files.insert(std::pair<int, boost::filesystem::path>(number, it->path()));
    }
  }
}

/** \brief Main entrance point of the script.
 * Expects one parameter, the path to the corresponding config file in config/.
 */
int main(int argc, char** argv) {
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("input",  boost::program_options::value<std::string>(), "input, either single OFF file or directory containing OFF files where the names correspond to integers (zero padding allowed) and are consecutively numbered starting with zero")
      ("reference", boost::program_options::value<std::string>(), "reference, either single OFF or TXT file or directory containing OFF or TXT files where the names correspond to integers (zero padding allowed) and are consecutively numbered starting with zero (the file names need to correspond to those found in the input directory); for TXT files, accuracy cannot be computed")
      ("output", boost::program_options::value<std::string>(), "output file, a TXT file containing accuracy and completeness for each input-reference pair as well as overall averages")
      ("n_points", boost::program_options::value<int>()->default_value(10000), "number points to sample from meshes in order to compute distances");

  boost::program_options::positional_options_description positionals;
  positionals.add("input", 1);
  positionals.add("reference", 1);
  positionals.add("output", 1);

  boost::program_options::variables_map parameters;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
  boost::program_options::notify(parameters);

  if (parameters.find("help") != parameters.end()) {
    std::cout << desc << std::endl;
    return 1;
  }

  boost::filesystem::path input(parameters["input"].as<std::string>());
  if (!boost::filesystem::is_directory(input) && !boost::filesystem::is_regular_file(input)) {
    std::cout << "Input is neither directory nor file." << std::endl;
    return 1;
  }

  boost::filesystem::path reference(parameters["reference"].as<std::string>());
  if (!boost::filesystem::is_directory(reference) && !boost::filesystem::is_regular_file(reference)) {
    std::cout << "Reference is neither directory nor file." << std::endl;
    return 1;
  }

  boost::filesystem::path output(parameters["output"].as<std::string>());
  if (boost::filesystem::is_regular_file(output)) {
    std::cout << "Output file already exists; overwriting." << std::endl;
  }

  int N_points = parameters["n_points"].as<int>();
  std::cout << "Using " << N_points << " points." << std::endl;

  std::map<int, boost::filesystem::path> input_files;
  std::map<int, boost::filesystem::path> reference_files;

  if (boost::filesystem::is_regular_file(input)) {
    if (input.extension().string() != ".off") {
      std::cout << "Only OFF files supported as input." << std::endl;
      return 1;
    }

    input_files.insert(std::pair<int, boost::filesystem::path>(0, input));
  }
  else {
    read_directory(input, input_files, {".off"});

    if (input_files.size() <= 0) {
      std::cout << "Could not find any OFF files in input directory." << std::endl;
      return 1;
    }

    std::cout << "Read " << input_files.size() << " input files." << std::endl;
  }

  if (boost::filesystem::is_regular_file(reference)) {
    if (reference.extension().string() != ".off" && reference.extension().string() != ".txt") {
      std::cout << "Only OFF or TXT files supported as reference." << std::endl;
      return 1;
    }

    reference_files.insert(std::pair<int, boost::filesystem::path>(0, reference));
  }
  else {
    read_directory(reference, reference_files, {".off", ".txt"});

    if (input_files.size() <= 0) {
      std::cout << "Could not find any OFF or TXT files in reference directory." << std::endl;
      return 1;
    }

    std::cout << "Read " << reference_files.size() << " reference files." << std::endl;
  }

  std::map<int, float> accuracies;
  std::map<int, float> completenesses;

  for (std::map<int, boost::filesystem::path>::iterator it = input_files.begin(); it != input_files.end(); it++) {

    int n = it->first;
    if (reference_files.find(n) == reference_files.end()) {
      std::cout << "Could not find the reference file corresponding to " << input_files[n] << "." << std::endl;
      return 1;
    }

    boost::filesystem::path input_file = input_files[n];
    boost::filesystem::path reference_file = reference_files[n];

    Mesh input_mesh;
    bool success = Mesh::from_off(input_file.string(), input_mesh);

    if (!success) {
      std::cout << "Could not read " << input_file << "." << std::endl;
      return 1;
    }

    if (reference_file.extension().string() == ".off") {
      Mesh reference_mesh;
      success = Mesh::from_off(reference_file.string(), reference_mesh);

      if (!success) {
        std::cout << "Could not read " << reference_file << "." << std::endl;
        return 1;
      }

      PointCloud input_point_cloud;
      printf("Sampling input mesh...");
      success = input_mesh.sample(N_points, input_point_cloud);


      printf("Computing accuracy ...");
      if (success) {
        float accuracy = 0;
        success = input_point_cloud.compute_distance(reference_mesh, accuracy);

        if (success) {
          accuracies[n] = accuracy;
          std::cout << "Computed accuracy for " << input_file << "." << std::endl;
        }
        else {
          std::cout << "Could not compute accuracy for " << input_file << "." << std::endl;
        }
      }
      else {
        std::cout << "Could not compute accuracy for " << input_file << "." << std::endl;
      }

      PointCloud reference_point_cloud;
      printf("Sampling reference mesh...");
      reference_mesh.sample(N_points, reference_point_cloud);


      printf("Computing completeness ...");
      if (success) {
        float completeness = 0;
        success = reference_point_cloud.compute_distance(input_mesh, completeness);

        if (success) {
          completenesses[n] = completeness;
          std::cout << "Computed completeness for " << input_file << "." << std::endl;
        }
        else {
          std::cout << "Could not compute completeness for " << input_file << "." << std::endl;
        }
      }
      else {
        std::cout << "Could not compute completeness for " << input_file << "." << std::endl;
      }
    }
    else if (reference_file.extension().string() == ".txt") {
      PointCloud reference_point_cloud;
      success = PointCloud::from_txt(reference_file.string(), reference_point_cloud);

      if (!success) {
        std::cout << "Could not read " << reference_file << "." << std::endl;
        return 1;
      }

      float completeness = 0;
      success = reference_point_cloud.compute_distance(input_mesh, completeness);

      if (success) {
        completenesses[n] = completeness;
        std::cout << "Computed completeness for " << input_file << "." << std::endl;
      }
      else {
        std::cout << "Could not compute completeness for " << input_file << "." << std::endl;
      }
    }
    else {
      std::cout << "Reference file " << reference_file << " has invalid extension." << std::endl;
    }
  }

  std::ofstream* out = new std::ofstream(output.string(), std::ofstream::out);
  if (!static_cast<bool>(*out)) {
    std::cout << "Could not open " << output << std::endl;
    exit(1);
  }

  float accuracy = 0;
  float completeness = 0;

  for (std::map<int, boost::filesystem::path>::iterator it = input_files.begin(); it != input_files.end(); it++) {
    int n = it->first;

    (*out) << n << " ";
    if (accuracies.find(n) != accuracies.end()) {
      (*out) << accuracies[n];
      accuracy += accuracies[n];
    }
    else {
      (*out) << "-1";
    }

    (*out) << " ";
    if (completenesses.find(n) != completenesses.end()) {
      (*out) << completenesses[n];
      completeness += completenesses[n];
    }
    else {
      (*out) << "-1";
    }

    (*out) << std::endl;
  }


  if (accuracies.size() > 0) {
    accuracy /= accuracies.size();
    (*out) << accuracy;
    std::cout << "Accuracy (input to reference): " << accuracy << std::endl;
  }
  else {
    (*out) << "-1";
    std::cout << "Could not compute accuracy." << std::endl;
  }

  (*out) << " ";
  if (completenesses.size() > 0) {
    completeness /= completenesses.size();
    (*out) << completeness;
  std::cout << "Completeness (reference to input): " << completeness << std::endl;
  }
  else {
    (*out) << "-1";
    std::cout << "Could not compute completeness." << std::endl;
  }

  out->close();
  delete out;
  std::cout << "Wrote " << output << "." << std::endl;

  exit(0);
}