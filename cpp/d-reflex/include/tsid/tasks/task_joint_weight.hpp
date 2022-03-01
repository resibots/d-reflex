//
// Copyright (c) 2017 CNRS
//
// This file is part of tsid
// tsid is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// tsid is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// tsid If not, see
// <http://www.gnu.org/licenses/>.
//

#ifndef __invdyn_task_joint_weight_hpp__
#define __invdyn_task_joint_weight_hpp__

#include "tsid/math/fwd.hpp"
#include "tsid/tasks/task-motion.hpp"
#include "tsid/trajectories/trajectory-base.hpp"
#include "tsid/math/constraint-equality.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace tsid
{
  namespace tasks
  {

    class TaskJointWeight : public TaskMotion
    {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      typedef math::Index Index;
      typedef trajectories::TrajectorySample TrajectorySample;
      typedef math::Vector Vector;
      typedef math::VectorXi VectorXi;
      typedef math::ConstraintEquality ConstraintEquality;
      typedef pinocchio::Data::Matrix6x Matrix6x;
      typedef pinocchio::Data Data;
      TaskJointWeight(const std::string & name,
                         RobotWrapper & robot);

      int dim() const;
      const Vector & mask() const;
      void mask(const Vector & mask);
      virtual void setMask(math::ConstRefVector mask);
      const ConstraintBase & compute(const double t,
                                     ConstRefVector q,
                                     ConstRefVector v,
                                     Data & data);

      const ConstraintBase & getConstraint() const;

      void set_weights(const Vector& weights);
      void setReference(const TrajectorySample & ref); 
      const Vector & Kp();
      const Vector & Kd();
      void Kp(ConstRefVector Kp);
      void Kd(ConstRefVector Kp);


    protected:
      Vector m_Kp;
      Vector m_Kd;
      Vector m_mask;
      Eigen::Matrix<double, 1, 1> m_C;
      Vector m_weights;
      Eigen::MatrixXd m_matrix;
      Vector m_vector; 
      //Vector m_p0;
      //double m_r0;
      //double m_coef;
      VectorXi m_activeAxes;
      Vector m_p_error, m_v_error;
      Vector m_p, m_v;
      Vector m_a_des;
      TrajectorySample m_ref;
      ConstraintEquality m_constraint;
    };

  }
}

#endif 
