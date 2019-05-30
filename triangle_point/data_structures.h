#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cfloat>

// Eigen
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

// Boost
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

// OpenMP
#include "omp.h"

#include "poitri.h"

/** \brief Compute triangle point distance and corresponding closest point.
 * \param[in] point point
 * \param[in] v1 first vertex
 * \param[in] v2 second vertex
 * \param[in] v3 third vertex
 * \param[out] ray corresponding closest point
 * \return distance
 */
float triangle_point_distance(const Eigen::Vector3f point,
  const Eigen::Vector3f v1, const Eigen::Vector3f v2,
  const Eigen::Vector3f v3,
  Eigen::Vector3f &closest_point);

/** \brief Point cloud class forward declaration. */
class PointCloud;

/** \brief Just encapsulating vertices and faces. */
class Mesh {
public:
  /** \brief Empty constructor. */
  Mesh() {

  }

  /** \brief Reading an off file and returning the vertices x, y, z coordinates and the
   * face indices.
   * \param[in] filepath path to the OFF file
   * \param[out] mesh read mesh with vertices and faces
   * \return success
   */
  static bool from_off(const std::string filepath, Mesh& mesh) {

    std::ifstream* file = new std::ifstream(filepath.c_str());
    std::string line;
    std::stringstream ss;
    int line_nb = 0;

    std::getline(*file, line);
    ++line_nb;

    if (line != "off" && line != "OFF") {
      std::cout << "[Error] Invalid header: \"" << line << "\", " << filepath << std::endl;
      return false;
    }

    size_t n_edges;
    std::getline(*file, line);
    ++line_nb;

    int n_vertices;
    int n_faces;
    ss << line;
    ss >> n_vertices;
    ss >> n_faces;
    ss >> n_edges;

    for (size_t v = 0; v < n_vertices; ++v) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      Eigen::Vector3f vertex;
      ss << line;
      ss >> vertex(0);
      ss >> vertex(1);
      ss >> vertex(2);

      mesh.add_vertex(vertex);
    }

    size_t n;
    for (size_t f = 0; f < n_faces; ++f) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      size_t n;
      ss << line;
      ss >> n;

      if(n != 3) {
        std::cout << "[Error] Not a triangle (" << n << " points) at " << (line_nb - 1) << std::endl;
        return false;
      }

      Eigen::Vector3i face;
      ss >> face(0);
      ss >> face(1);
      ss >> face(2);

      mesh.add_face(face);
    }

    if (n_vertices != mesh.num_vertices()) {
      std::cout << "[Error] Number of vertices in header differs from actual number of vertices." << std::endl;
      return false;
    }

    if (n_faces != mesh.num_faces()) {
      std::cout << "[Error] Number of faces in header differs from actual number of faces." << std::endl;
      return false;
    }

    file->close();
    delete file;

    return true;
  }

  /** \brief Write mesh to OFF file.
   * \param[in] filepath path to OFF file to write
   * \return success
   */
  bool to_off(const std::string filepath) const {
    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(out)) {
      return false;
    }

    (*out) << "OFF" << std::endl;
    (*out) << this->num_vertices() << " " << this->num_faces() << " 0" << std::endl;

    for (unsigned int v = 0; v < this->num_vertices(); v++) {
      (*out) << this->vertices[v](0) << " " << this->vertices[v](1) << " " << this->vertices[v](2) << std::endl;
    }

    for (unsigned int f = 0; f < this->num_faces(); f++) {
      (*out) << "3 " << this->faces[f](0) << " " << this->faces[f](1) << " " << this->faces[f](2) << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

  /** \brief Add a vertex.
   * \param[in] vertex vertex to add
   */
  void add_vertex(Eigen::Vector3f& vertex) {
    this->vertices.push_back(vertex);
  }

  /** \brief Get the number of vertices.
   * \return number of vertices
   */
  int num_vertices() const {
    return static_cast<int>(this->vertices.size());
  }

  /** \brief Get a vertex.
   * \param[in] v vertex index
   * \return vertex
   */
  Eigen::Vector3f vertex(int v) const {
    assert(v >= 0 && v < this->num_vertices());
    return this->vertices[v];
  }

  /** \brief Add a face.
   * \param[in] face face to add
   */
  void add_face(Eigen::Vector3i& face) {
    this->faces.push_back(face);
  }

  /** \brief Get the number of faces.
   * \return number of faces
   */
  int num_faces() const {
    return static_cast<int>(this->faces.size());
  }

  /** \brief Get a face.
   * \param[in] f face index
   * \return face
   */
  Eigen::Vector3i face(int f) const {
    assert(f >= 0 && f < this->num_faces());
    return this->faces[f];
  }

  /** \brief Sample points from the mesh
   * \param[in] mesh mesh to sample from
   * \param[in] n batch index in points
   * \param[in] points pre-initialized tensor holding points
   */
  bool sample(const int N, PointCloud &point_cloud) const;

private:

  /** \brief Vertices as (x,y,z)-vectors. */
  std::vector<Eigen::Vector3f> vertices;

  /** \brief Faces as list of vertex indices. */
  std::vector<Eigen::Vector3i> faces;
};

/** \brief Class representing a point cloud in 3D. */
class PointCloud {
public:
  /** \brief Constructor. */
  PointCloud() {

  }

  /** \brief Copy constructor.
   * \param[in] point_cloud point cloud to copy
   */
  PointCloud(const PointCloud &point_cloud) {
    this->points.clear();

    for (unsigned int i = 0; i < point_cloud.points.size(); i++) {
      this->points.push_back(point_cloud.points[i]);
    }
  }

  /** \brief Destructor. */
  ~PointCloud() {

  }

  /** \brief Read point cloud from txt file.
   * \param[in] filepath path to file to read
   * \param[out] point_cloud
   * \return success
   */
  static bool from_txt(const std::string &filepath, PointCloud &point_cloud) {
    std::ifstream file(filepath.c_str());
    std::string line;
    std::stringstream ss;

    std::getline(file, line);
    ss << line;

    int n_points = 0;
    ss >> n_points;

    if (n_points < 0) {
      return false;
    }

    for (int i = 0; i < n_points; i++) {
      std::getline(file, line);

      ss.clear();
      ss.str("");
      ss << line;

      Eigen::Vector3f point(0, 0, 0);
      ss >> point(0);
      ss >> point(1);
      ss >> point(2);

      point_cloud.add_point(point);
    }

    return true;
  }

  /** \brief Add a point to the point cloud.
   * \param[in] point point to add
   */
  void add_point(const Eigen::Vector3f &point) {
    this->points.push_back(point);
  }

  /** \brief Get number of points.
   * \return number of points
   */
  unsigned int num_points() const {
    return this->points.size();
  }

  /** \brief Compute distance to mesh.
    * \param[in] mesh
    * \param[out] distances per point distances
    * \param[out] distance
    * \return success
    */
  bool compute_distance(const Mesh &mesh, float &_distance) {
    _distance = 0;

    if (this->num_points() <= 0) {
      std::cout << "[Error] no points in this point clouds" << std::endl;
      return false;
    }

    if (mesh.num_faces() <= 0) {
      std::cout << "[Error] no faces in given mesh" << std::endl;
      return false;
    }

    #pragma omp parallel
    {
      #pragma omp for
      for (unsigned int i = 0; i < this->points.size(); i++) {

        float min_distance = FLT_MAX;
        for (int f = 0; f < mesh.num_faces(); f++) {
          Eigen::Vector3f closest_point;
          Eigen::Vector3f v1 = mesh.vertex(mesh.face(f)(0));
          Eigen::Vector3f v2 = mesh.vertex(mesh.face(f)(1));
          Eigen::Vector3f v3 = mesh.vertex(mesh.face(f)(2));

          triangle_point_distance(this->points[i], v1, v2, v3, closest_point);
          float distance = (this->points[i] - closest_point).norm();

          if (distance < min_distance) {
            min_distance = distance;
          }
        }

        #pragma omp atomic
        _distance += min_distance;
      }
    }

    _distance /= this->num_points();
    return true;
  }

private:
  /** \brief The points of the point cloud. */
  std::vector<Eigen::Vector3f> points;

};
