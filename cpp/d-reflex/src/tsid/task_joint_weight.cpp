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

//#define DEBUG_POS_AVOIDANCE

#include "tsid/tasks/task_joint_weight.hpp"
#include "tsid/robots/robot-wrapper.hpp"

namespace tsid
{
  namespace tasks
  {
    using namespace math;
    using namespace trajectories;
    using namespace pinocchio;

    TaskJointWeight::TaskJointWeight(const std::string & name,
                                           RobotWrapper & robot):
      TaskMotion(name, robot),
      m_constraint(name, robot.na(), robot.nv())
    {
      m_Kp.setZero(robot.na());
      m_Kd.setZero(robot.na());
      m_weights.setZero(robot.na());
      m_matrix = Eigen::MatrixXd::Zero(m_robot.na(), m_robot.nv()); 
      m_vector.setZero(robot.nv()); 
      Vector m = Vector::Ones(robot.na());
      //setMask(m);
    }

    int TaskJointWeight::dim() const
    {
      return (int)m_mask.sum();
    }

    const Vector & TaskJointWeight::mask() const
    {
      return m_mask;
    }

    void TaskJointWeight::mask(const Vector & m)
    {
      assert(m.size()==m_robot.na());
      m_mask = m;
    }

    void TaskJointWeight::setMask(ConstRefVector m)
    {
      assert(m.size()==m_robot.na());
      m_mask = m;
      const Vector::Index dim = static_cast<Vector::Index>(m.sum());
      Matrix S = Matrix::Zero(dim, m_robot.nv());
      m_activeAxes.resize(dim);
      unsigned int j=0;
      for(unsigned int i=0; i<m.size(); i++)
        if(m(i)!=0.0)
        {
          assert(m(i)==1.0);
          S(j,m_robot.nv()-m_robot.na()+i) = 1.0;
          m_activeAxes(j) = i;
          j++;
        }
      m_constraint.resize((unsigned int)dim, m_robot.nv());
      m_constraint.setMatrix(S);
    }

    const Vector & TaskJointWeight::Kp(){ return m_Kp; }

    const Vector & TaskJointWeight::Kd(){ return m_Kd; }

    void TaskJointWeight::Kp(ConstRefVector Kp)
    {
      assert(Kp.size()==m_robot.na());
      m_Kp = Kp;
    }

    void TaskJointWeight::Kd(ConstRefVector Kd)
    {
      assert(Kd.size()==m_robot.na());
      m_Kd = Kd;
    }

    const ConstraintBase & TaskJointWeight::getConstraint() const
    {
      return m_constraint;
    }
    
    void TaskJointWeight::set_weights(const Vector& weights)
    {
      m_weights = weights;
      for (uint i=6; i<m_robot.nv(); i++){
        m_matrix(i-6,i) = m_weights[i-6];
      }
    }

    void TaskJointWeight::setReference(const TrajectorySample & ref)
    {
      assert(ref.pos.size()==m_robot.na());
      assert(ref.vel.size()==m_robot.na());
      assert(ref.acc.size()==m_robot.na());
      m_ref = ref;
    }

    const ConstraintBase & TaskJointWeight::compute(const double,
                                                    ConstRefVector q,
                                                    ConstRefVector v,
                                                    Data & data)
    {
      // Set A
      m_constraint.setMatrix(m_matrix);
      /*
      double dt = 0.001;
      m_v = v.tail(m_robot.na());
      m_constraint.setVector(-2*m_v/dt);
      */
      // Compute errors
      m_p = q.tail(m_robot.na());
      m_v = v.tail(m_robot.na());
      m_p_error = m_p - m_ref.pos;
      m_v_error = m_v - m_ref.vel;
      m_a_des = - m_Kp.cwiseProduct(m_p_error)
                - m_Kd.cwiseProduct(m_v_error)
                + m_ref.acc;

      for(unsigned int i=0; i<m_robot.na(); i++)
        m_constraint.vector()(i) = m_weights(i) * m_a_des(i);
      
      return m_constraint;
    }
  }
}
