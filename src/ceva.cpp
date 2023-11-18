#include <iostream>
#include <string>

#include <fstream>
#include <boost/shared_ptr.hpp>
#include <omp.h>
#include <thread>

#include <Eigen/Dense>

#include "basalt/spline/se3_spline.h"
#include "basalt/spline/posesplinex.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "factor/PoseAnalyticFactor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

// Shorthand for print color
#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"
#define RESET "\033[0m"

// Shortened typedef matching character length of Vector3d and Matrix3d
typedef Eigen::Quaterniond Quaternd;
typedef Eigen::Quaterniond Quaternf;

// Shorthand for sophus objects
typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;

#define MAX_THREADS std::thread::hardware_concurrency()

void ComputeCeresCost(vector<ceres::internal::ResidualBlock *> &res_ids,
                      double &cost, ceres::Problem &problem)
{
    if (res_ids.size() == 0)
    {
        cost = -1;
        return;
    }

    ceres::Problem::EvaluateOptions e_option;
    e_option.residual_blocks = res_ids;
    e_option.num_threads = omp_get_max_threads();
    problem.Evaluate(e_option, &cost, NULL, NULL, NULL);
}

// Load deliminated file to matrix with unknown size
template <typename Scalar = double, int RowSize = Dynamic, int ColSize = Dynamic>
Matrix<Scalar, RowSize, ColSize> load_dlm(const std::string &path, string dlm, int r_start = 0, int col_start = 0)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int row_idx = -1;
    int rows = 0;
    while (std::getline(indata, line))
    {
        row_idx++;
        if (row_idx < r_start)
            continue;

        std::stringstream lineStream(line);
        std::string cell;
        int col_idx = -1;
        while (std::getline(lineStream, cell, dlm[0]))
        {
            if (cell == dlm || cell.size() == 0)
                continue;

            col_idx++;
            if (col_idx < col_start)
                continue;

            values.push_back(std::stod(cell));
        }

        rows++;
    }

    return Map<const Matrix<Scalar, RowSize, ColSize, RowMajor>>(values.data(), rows, values.size() / rows);
}


inline std::string myprintf(const std::string& format, ...)
{
    va_list args;
    va_start(args, format);
    size_t len = std::vsnprintf(NULL, 0, format.c_str(), args);
    va_end(args);
    std::vector<char> vec(len + 1);
    va_start(args, format);
    std::vsnprintf(&vec[0], len + 1, format.c_str(), args);
    va_end(args);
    
    return string(vec.begin(), vec.end() - 1);
}

class Ceva
{
private:
    using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
    PoseSplinePtr traj;

public:
    // Simple destructor
    ~Ceva(){};

    // Simple constructor
    Ceva(){};

    // Constructor using order and knot length
    Ceva(int N, double dt)
    {
        traj = std::make_shared<PoseSplineX>(PoseSplineX(N, dt));
        printf("Created spline of order %d and knot length %f\n", N, traj->numKnots(), dt);
    }

    // Constructor using start time and final time
    Ceva(int N, double dt, double start_time, double final_time)
    {
        traj = std::make_shared<PoseSplineX>(PoseSplineX(N, dt));
        traj->setStartTime(start_time);
        traj->extendKnotsTo(final_time, Sophus::SE3<double>(Quaternd::Identity(), Vector3d(0, 0, 0)));

        printf("Created spline of order %d, knot length %f, from time %f to %f; consisting of %d knots\n",
                N, dt, traj->minTime(), traj->maxTime(), traj->numKnots());
    }

    // Constructor using log file
    Ceva(int N, double dt, double start_time, string spline_log)
    {
        // Read the spline log
        MatrixXd knots_pose = load_dlm<double, Dynamic, Dynamic>(spline_log, ",", 1, 0);
        
        int num_knots = knots_pose.rows();
        int order = N;
        
        printf("Creating spline of order %d and %d knots from log file %s\n",
                order, num_knots, spline_log.c_str());

        traj = std::make_shared<PoseSplineX>(PoseSplineX(order, dt));
        double final_time = start_time + (num_knots - order  + 1)*dt;
        
        // Set the start and end time
        traj->setStartTime(start_time);
        traj->extendKnotsTo(final_time, Sophus::SE3<double>(Quaternd::Identity(), Vector3d(0, 0, 0)));

        // Make sure the number of knots inferred from final time matches with the log
        assert(num_knots == traj->numKnots());

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < traj->numKnots(); i++)
        {
            // assert( (int)knots_pose(i, 0) == i );
            // assert( fabs(knots_pose(i, 1) - traj->getKnotTime(i)) < 1e-3 );

            Vector3d p(knots_pose(i, 2), knots_pose(i, 3), knots_pose(i, 4));
            Quaternd q(knots_pose(i, 8), knots_pose(i, 5), knots_pose(i, 6), knots_pose(i, 7));

            // printf("Setting knot %5d with pose: %.3f. %.3f. %.3f. %.3f. %.3f. %.3f. %.3f\n",
            //         i,
            //         p.x(), p.y(), p.z(),
            //         q.x(), q.y(), q.z(), q.w());

            traj->setKnot(Sophus::SE3<double>(q, p), i);
        }

        printf("Done\n");
    }

    // Constructor using matrices
    Ceva(int N, double dt, double start_time, MatrixXd &knots_pose)
    {
        int num_knots = knots_pose.rows();
        int order = N;

        printf("Creating spline of order %d and %d knots\n", order, num_knots);

        traj = std::make_shared<PoseSplineX>(PoseSplineX(N, dt));
        double final_time = start_time + (num_knots - order + 1) * dt;

        traj->setStartTime(start_time);
        traj->extendKnotsTo(final_time, Sophus::SE3<double>(Quaternd::Identity(), Vector3d(0, 0, 0)));

        // Assert the number of knots min log and calculating being equal
        assert(num_knots == traj->numKnots());

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < traj->numKnots(); i++)
        {
            // assert( (int)knots_pose(i, 0) == i );
            // assert( fabs(knots_pose(i, 1) - traj->getKnotTime(i)) < 1e-6 );
            // assert( fabs(knots_pose(i, 1) - traj->getKnotTime(i)) < 1e-6 &&
            //         "Log time: %f. Knot time: %f\n",
            //         knots_pose(i, 1), traj->getKnotTime(i));

            Vector3d p(knots_pose(i, 0), knots_pose(i, 1), knots_pose(i, 2));
            Quaternd q(knots_pose(i, 6), knots_pose(i, 3), knots_pose(i, 4), knots_pose(i, 5));

            // printf("Setting knot %5d with pose: %.3f. %.3f. %.3f. %.3f. %.3f. %.3f. %.3f\n",
            //         i,
            //         p.x(), p.y(), p.z(),
            //         q.x(), q.y(), q.z(), q.w());

            traj->setKnot(Sophus::SE3<double>(q, p), i);
        }

        printf("Done\n");
    }

    // Get the pose at sample times
    MatrixXd getPose(vector<double> sample_times)
    {
        MatrixXd pose(sample_times.size(), 8);

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < sample_times.size(); i++)
        {
            double t = sample_times[i];

            if (t < traj->minTime() + 1e-6)
            {
                // printf("Sample time %.3f is before start_time %.3f\n", t, traj->minTime());
                t = traj->minTime() + 1e-6;
            }
            else if (t >= traj->maxTime() - 1e-6)
            {
                // printf("Sample time %.3f is after final_time %.3f\n", t, traj->maxTime());
                t = traj->maxTime() - 1e-6;
            }
            // else
            //     printf("Sample time %.3f is in range final_time %.3f\n", t, traj->maxTime());

            Sophus::SE3<double> se3 = traj->pose(t);
            Vector3d p = se3.translation();
            Quaternd q = se3.so3().unit_quaternion();

            pose.row(i) << t, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w();
        }

        return pose;
    }

    MatrixXd getPose(double sample_time)
    {
        return getPose(vector<double>(1, sample_time));
    }

    MatrixXd getKnotPose(vector<int> knot_idx)
    {
        MatrixXd pose(knot_idx.size(), 8);

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int i = 0; i < knot_idx.size(); i++)
        {
            double   t = traj->getKnotTime(knot_idx[i]);
            Quaternd q = traj->getKnotSO3(knot_idx[i]).unit_quaternion();
            Vector3d p = traj->getKnotPos(knot_idx[i]);

            pose.row(i) << t, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w();
        }

        return pose;
    }

    MatrixXd getKnotPose(int knot_idx)
    {
        return getKnotPose(vector<int>(1, knot_idx));
    }

    MatrixXd getAllKnotPose()
    {
        vector<int> knot_idx(traj->numKnots());
        for(int i = 0; i < traj->numKnots(); i++)
            knot_idx[i]= i;
        return getKnotPose(knot_idx);
    }

    MatrixXd getAngularVelWorld(vector<double> sample_times)
    {
        MatrixXd omega(sample_times.size(), 4);

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < sample_times.size(); i++)
        {
            double t = sample_times[i];

            if (t < traj->minTime() + 1e-6)
            {
                // printf("Sample time %.3f is before start_time %.3f\n", t, traj->minTime());
                t = traj->minTime() + 1e-6;
            }
            else if (t >= traj->maxTime() - 1e-6)
            {
                // printf("Sample time %.3f is after final_time %.3f\n", t, traj->maxTime());
                t = traj->maxTime() - 1e-6;
            }
            // else
            //     printf("Sample time %.3f is in range final_time %.3f\n", t, traj->maxTime());

            Vector3d omegainW = traj->pose(t).so3().unit_quaternion() * traj->rotVelBody(t);
            omega.row(i) << t, omegainW.x(), omegainW.y(), omegainW.z();

            // std::cout << omega << std::endl;
        }

        return omega;
    }

    MatrixXd getAngularVelWorld(double sample_time)
    {
        return getAngularVelWorld(vector<double>(1, sample_time));
    }

    MatrixXd getTransVelWorld(vector<double> sample_times)
    {
        MatrixXd vel(sample_times.size(), 4);

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < sample_times.size(); i++)
        {
            double t = sample_times[i];

            if (t < traj->minTime() + 1e-6)
            {
                // printf("Sample time %.3f is before start_time %.3f\n", t, traj->minTime());
                t = traj->minTime() + 1e-6;
            }
            else if (t >= traj->maxTime() - 1e-6)
            {
                // printf("Sample time %.3f is after final_time %.3f\n", t, traj->maxTime());
                t = traj->maxTime() - 1e-6;
            }
            // else
            //     printf("Sample time %.3f is in range final_time %.3f\n", t, traj->maxTime());

            Vector3d v = traj->transVelWorld(t);

            vel.row(i) << t, v.x(), v.y(), v.z();
        }

        return vel;
    }

    MatrixXd getTransVelWorld(double sample_time)
    {
        return getTransVelWorld(vector<double>(1, sample_time));
    }

    MatrixXd getTransAccelWorld(vector<double> sample_times)
    {
        MatrixXd accel(sample_times.size(), 4);

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < sample_times.size(); i++)
        {
            double t = sample_times[i];

            if (t < traj->minTime() + 1e-6)
            {
                // printf("Sample time %.3f is before start_time %.3f\n", t, traj->minTime());
                t = traj->minTime() + 1e-6;
            }
            else if (t >= traj->maxTime() - 1e-6)
            {
                // printf("Sample time %.3f is after final_time %.3f\n", t, traj->maxTime());
                t = traj->maxTime() - 1e-6;
            }
            // else
            //     printf("Sample time %.3f is in range final_time %.3f\n", t, traj->maxTime());

            Vector3d a = traj->transAccelWorld(t);

            accel.row(i) << t, a.x(), a.y(), a.z();
        }

        return accel;
    }

    MatrixXd getTransAccelWorld(double sample_time)
    {
        return getTransAccelWorld(vector<double>(1, sample_time));
    }

    MatrixXd blendingMatrix(bool cummulative = false)
    {
        return traj->blendingMatrix(cummulative);
    }

    int order()
    {
        return traj->order();
    }

    double getDt()
    {
        return traj->getDt();
    }

    double minTime() const
    {        
        return traj->minTime();
    }

    double maxTime() const
    {
        return traj->maxTime();
    }

    int numKnots()
    {
        return traj->numKnots();
    }
    
    // Generate a random trajectory
    void genRandomTrajectory(int n)
    {
        traj->genRandomTrajectory(n);
    }
    
    // Deskew a pointcloud
    MatrixXd deskewCloud(MatrixXd p_inBs, vector<double> tpoint)
    {
        // Number of points
        int Npt = p_inBs.rows();

        // Create output points
        MatrixXd p_inW(Npt, 3);

        // Transform the points
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < Npt; i++)
        {
            double t = tpoint[i];

            if (t < traj->minTime() + 1e-6)
            {
                // printf("Sample time %.3f is before start_time %.3f\n", t, traj->minTime());
                t = traj->minTime() + 1e-6;
            }
            else if (t >= traj->maxTime() - 1e-6)
            {
                // printf("Sample time %.3f is after final_time %.3f\n", t, traj->maxTime());
                t = traj->maxTime() - 1e-6;
            }

            Sophus::SE3<double> se3 = traj->pose(t);
            p_inW.block<1, 3>(i, 0) = (se3.so3().unit_quaternion() * p_inBs.block<1, 3>(i, 0).transpose() + se3.translation()).transpose();
        }

        return p_inW;
    }

    // Fit the spline with data
    string fitspline(vector<double> ts, MatrixXd pos, MatrixXd rot, vector<double> wp, vector<double> wr, double loss_thres)
    {
        // Create spline
        int KNOTS = traj->numKnots();
        int N = traj->order();

        // Ceres problem
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        // Set up the ceres problem
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 10;

        ceres::LossFunction *loss_function = new ceres::CauchyLoss(loss_thres);

        ceres::LocalParameterization *local_parameterization = new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>();

        // Add the parameter blocks for rotation
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj->getKnotSO3(knot_idx).data(), 4, local_parameterization);

        // Add the parameter blocks for position
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj->getKnotPos(knot_idx).data(), 3);

        double cost_pose_init;
        double cost_pose_final;
        vector<ceres::internal::ResidualBlock *> res_ids_pose;
        // Add the pose factors
        for (int k = 0; k < ts.size(); k++)
        {
            double t = ts[k];

            // Continue if sample is in the window
            if (t < traj->minTime() + 1e-6 && t > traj->maxTime() - 1e-6)
                continue;

            auto us = traj->computeTIndex(t);
            double u = us.first;
            int s = us.second;

            Quaternd q(rot(k, 3), rot(k, 0), rot(k, 1), rot(k, 2));
            Vector3d p(pos(k, 0), pos(k, 1), pos(k, 2));

            ceres::CostFunction *cost_function = new PoseAnalyticFactor(SE3d(q, p), wp[k], wr[k], N, traj->getDt(), u);

            // Find the coupled poses
            vector<double *> factor_param_blocks;
            for (int knot_idx = s; knot_idx < s + N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotSO3(knot_idx).data());

            for (int knot_idx = s; knot_idx < s + N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotPos(knot_idx).data());

            // // printf("Creating functor: u: %f, s: %d. sample: %d / %d\n", u, s, i, pose_gt.size());
            // // cost_function->SetNumResiduals(6);

            auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }

        // Init cost
        ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

        // Solve the optimization problem
        ceres::Solve(options, &problem, &summary);

        // Final cost
        ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

        string report = myprintf("Spline Fitting. Cost: %f -> %f. Iterations: %d.\n",
                                 cost_pose_init, cost_pose_final, summary.iterations.size());

        return report;
    }
};

PYBIND11_MODULE(ceva, m)
{
    py::class_<Ceva>(m, "Ceva")

        /* #region Constructor --------------------------------------------------------------------------------------*/

        .def(py::init<>())
        .def(py::init<const int &, const double &>())
        .def(py::init<const int &, const double &, const double &, const double &>())
        .def(py::init<const int &, const double &, const double &, const string &>())
        .def(py::init<const int &, const double &, const double &, MatrixXd &>())

        /* #endregion Constructor -----------------------------------------------------------------------------------*/


        /* #region Get information from spline ----------------------------------------------------------------------*/

        // Some fundamental parameters
        .def("blendingMatrix", static_cast<MatrixXd (Ceva::*)(bool)>(&Ceva::blendingMatrix), "Get the blending matrix of the spline.")
        .def("minTime", &Ceva::minTime, "Get the spline minimum time.")
        .def("maxTime", &Ceva::maxTime, "Get the spline maximum time.")
        .def("numKnots", &Ceva::numKnots, "Get the number of knots of the spline.")
        .def("order", &Ceva::order, "Get the order of the spline")
        .def("getDt", &Ceva::getDt, "Get the knot length of the spline.")

        // Derivative information
        .def("getPose", static_cast<MatrixXd (Ceva::*)(vector<double>)>(&Ceva::getPose), "Get the poses and multiple times.")
        .def("getPose", static_cast<MatrixXd (Ceva::*)(double)>(&Ceva::getPose), "Get the pose at one time.")

        .def("getKnotPose", static_cast<MatrixXd (Ceva::*)(vector<int>)>(&Ceva::getKnotPose), "Get the knot poses by a list of index.")
        .def("getKnotPose", static_cast<MatrixXd (Ceva::*)(int)>(&Ceva::getKnotPose), "Get the knot pose at one time.")
        .def("getAllKnotPose", static_cast<MatrixXd (Ceva::*)()>(&Ceva::getAllKnotPose), "Get all the knot poses.")

        .def("getAngularVelWorld", static_cast<MatrixXd (Ceva::*)(vector<double>)>(&Ceva::getAngularVelWorld), "Get the angular velocity in inertial frame at multiple times.")
        .def("getAngularVelWorld", static_cast<MatrixXd (Ceva::*)(double)>(&Ceva::getAngularVelWorld), "Get the angular velocity in inertial frame at a single time.")

        .def("getTransVelWorld", static_cast<MatrixXd (Ceva::*)(vector<double>)>(&Ceva::getTransVelWorld), "Get the translational velocity in inertial frame at multiple times.")
        .def("getTransVelWorld", static_cast<MatrixXd (Ceva::*)(double)>(&Ceva::getTransVelWorld), "Get the translational velocity in inertial frame at a single time.")

        .def("getTransAccelWorld", static_cast<MatrixXd (Ceva::*)(vector<double>)>(&Ceva::getTransAccelWorld), "Get the translational acceleration in inertial frame at multiple times.")
        .def("getTransAccelWorld", static_cast<MatrixXd (Ceva::*)(double)>(&Ceva::getTransAccelWorld), "Get the translational acceleration in inertial frame at a single time.")

        /* #endregion Get information from spline -------------------------------------------------------------------*/


        /* #region Do something with the spline ---------------------------------------------------------------------*/
        
        .def("genRandomTrajectory", &Ceva::genRandomTrajectory, "Random initialization of the spline.")
        .def("fitspline", static_cast<string (Ceva::*)(vector<double>, MatrixXd, MatrixXd, vector<double>, vector<double>, double)>(&Ceva::fitspline), "Get a spline that best fits sampling data.")
        .def("deskewCloud", static_cast<MatrixXd (Ceva::*)(MatrixXd, vector<double>)>(&Ceva::deskewCloud), "Deskew pointcloud given the time stamps.")

        /* #endregion Do something with the spline -----------------------------------------------------------------*/
        
        ;
}
