/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SEn.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"

namespace esekfom {

using namespace Eigen;

template<typename T>
struct dyn_share_modified
{
	bool valid;
	bool converge;
	T M_Noise;
	Eigen::Matrix<T, Eigen::Dynamic, 1> R_vec; // per-point relative confidence weight w_j in (0,1]; empty => fall back to scalar M_Noise
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, 6, 1> z_IMU;
	Eigen::Matrix<T, 6, 1> R_IMU;
	bool satu_check[6];
};

template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF, m = state::DIM, l = measurement::DOF
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

	typedef void measurementModel_dyn_share_modified_cov(state &, Eigen::Matrix3d, Eigen::Matrix3d, dyn_share_modified<scalar_type> &);
	typedef void measurementModel_dyn_share_modified(state &, dyn_share_modified<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){};

	void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified_cov h_dyn_share_in1)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in1;
		// h_dyn_share_modified_3 = h_dyn_share_in3;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}
	
	void init_dyn_share_modified_3h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified_cov h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in1;
		h_dyn_share_modified_2 = h_dyn_share_in2;
		// h_dyn_share_modified_3 = h_dyn_share_in3;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}

	// iterated error state EKF propogation
	void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov){
		if (predict_state)
		{
			flatted_state f_ = f(x_, i_in);
			x_.oplus(f_, dt);
		}

		if (prop_cov)
		{
			flatted_state f_ = f(x_, i_in);
			// state x_before = x_;

			cov_ f_x_ = f_x(x_, i_in);
			cov f_x_final;
			F_x1 = cov::Identity();
			for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
				int idx = (*it).first.first;
				int dim = (*it).first.second;
				int dof = (*it).second;
				for(int i = 0; i < n; i++){
					for(int j=0; j<dof; j++)
					{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
				}
			}

			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;
				int dim = (*it).second;
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = -1 * f_(dim + i) * dt;
				}
				// MTK::SO3<scalar_type> res;
				// res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
				F_x1.template block<3, 3>(idx, idx) = MTK::SO3<scalar_type>::exp(seg_SO3); // res.normalized().toRotationMatrix();	 	
				res_temp_SO3 = MTK::A_matrix(seg_SO3);
				for(int i = 0; i < n; i++){
					f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
				}
			}
	
			F_x1 += f_x_final * dt;
			P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
		}
	}

	bool update_iterated_dyn_share_modified() {
		dyn_share_modified<scalar_type> dyn_share;
		state x_propagated = x_;
		int dof_Measurement;
		double m_noise;
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
			h_dyn_share_modified_1(x_, P_.template block<3, 3>(0, 0), P_. template block<3, 3>(3, 3), dyn_share);
			if(! dyn_share.valid)
			{
				return false;
				// continue;
			}
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
			// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R; 
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
			// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
			dof_Measurement = h_x.rows();
			m_noise = dyn_share.M_Noise;
			// per-point relative confidence weight w_j (in (0,1]); when sized to the measurements
			// it modulates each point's influence. w_j == 1 reproduces the scalar-noise behaviour exactly.
			Matrix<scalar_type, Eigen::Dynamic, 1> w_vec = dyn_share.R_vec;
			bool use_w = (w_vec.size() == dof_Measurement);

			// Accumulate the LiDAR pose information matrix (6x6) across the per-point updates so that
			// degeneracy can be assessed over a window rather than from a single rank-deficient update.
			if (degen_en_)
			{
				Matrix<scalar_type, Eigen::Dynamic, 6> Hp = h_x.block(0, 0, dof_Measurement, 6);
				Matrix<scalar_type, 6, 6> Mp;
				if (use_w) Mp = Hp.transpose() * w_vec.asDiagonal() * Hp;
				else       Mp = Hp.transpose() * Hp;
				degen_info_ = degen_decay_ * degen_info_ + Mp;
				degen_count_++;
			}
			// dof_Measurement_noise = dyn_share.R.rows();
			// vectorized_state dx, dx_new;
			// x_.boxminus(dx, x_propagated);
			// dx_new = dx;
			// P_ = P_propagated;

			Matrix<scalar_type, n, Eigen::Dynamic> PHT;
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
			Matrix<scalar_type, n, Eigen::Dynamic> K_;
			if(n > dof_Measurement)
			{
				PHT = P_. template block<n, 12>(0, 0) * h_x.transpose();
				HPHT = h_x * PHT.topRows(12);
				for (int m = 0; m < dof_Measurement; m++)
				{
					// inflate the variance of less-trustworthy points: R_j = m_noise / w_j
					HPHT(m, m) += use_w ? (m_noise / w_vec(m)) : m_noise;
				}
				K_= PHT*HPHT.inverse();
			}
			else
			{
				// information form: H^T R^{-1} H, with per-point information scaled by w_j
				Matrix<scalar_type, 12, 12> HTH;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HT_Rinv; // 12 x dof_Measurement
				if (use_w)
				{
					HT_Rinv = m_noise * h_x.transpose() * w_vec.asDiagonal();
				}
				else
				{
					HT_Rinv = m_noise * h_x.transpose();
				}
				HTH = HT_Rinv * h_x;
				Matrix<scalar_type, n, n> P_inv = P_.inverse();
				P_inv.template block<12, 12>(0, 0) += HTH;
				P_inv = P_inv.inverse();
				K_ = P_inv.template block<n, 12>(0, 0) * HT_Rinv;
			}

			// Degeneracy-aware gain suppression: project the pose rows of the Kalman gain onto the
			// well-constrained eigen-subspace of the accumulated information matrix, so degenerate
			// directions keep the kinematic/IMU prediction instead of being driven by weak LiDAR
			// constraints. Both mean and covariance updates use the projected gain => consistent.
			// The translation (rows 0-2) and rotation (rows 3-5) blocks are analysed SEPARATELY:
			// the rotation Jacobian carries a lever-arm (point range), so its information is orders
			// of magnitude larger than translation's; mixing them in one 6x6 ratio test would always
			// flag translation as degenerate. Each block is compared against its own lambda_max.
			if (degen_en_ && degen_count_ > degen_warmup_)
			{
				for (int blk = 0; blk < 2; blk++) // 0: translation (rows 0-2), 1: rotation (rows 3-5)
				{
					const int off = blk * 3;
					Eigen::SelfAdjointEigenSolver<Matrix<scalar_type, 3, 3> > es(degen_info_.template block<3, 3>(off, off));
					const Matrix<scalar_type, 3, 1> eval = es.eigenvalues();   // ascending order
					const Matrix<scalar_type, 3, 3> evec = es.eigenvectors();
					scalar_type lmax = eval(2);
					if (lmax <= scalar_type(1e-9)) continue;
					Matrix<scalar_type, 3, 3> Pf = Matrix<scalar_type, 3, 3>::Zero();
					bool any_degen = false;
					for (int e = 0; e < 3; e++)
					{
						if (eval(e) / lmax >= degen_ratio_thr_) Pf += evec.col(e) * evec.col(e).transpose();
						else any_degen = true;
					}
					if (any_degen)
					{
						Matrix<scalar_type, 3, Eigen::Dynamic> Kblk = Pf * K_.block(off, 0, 3, dof_Measurement);
						K_.block(off, 0, 3, dof_Measurement) = Kblk;
					}
				}
			}

			Matrix<scalar_type, n, 1> dx_ = K_ * z; // - h) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
			// state x_before = x_;

			x_.boxplus(dx_);
			{
				P_ = P_ - K_*h_x*P_. template block<12, n>(0, 0);
			}
		}
		return true;
	}
	
	void update_iterated_dyn_share_IMU() {
		
		dyn_share_modified<scalar_type> dyn_share;
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
			h_dyn_share_modified_2(x_, dyn_share);

			Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;

			Matrix<double, 30, 6> PHT;
            Matrix<double, 6, 30> HP;
            Matrix<double, 6, 6> HPHT;
			PHT.setZero();
			HP.setZero();
			HPHT.setZero();
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
					PHT.col(l_) = P_.col(15+l_) + P_.col(24+l_);
					HP.row(l_) = P_.row(15+l_) + P_.row(24+l_);
				}
			}
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
					HPHT.col(l_) = HP.col(15+l_) + HP.col(24+l_);
				}
				HPHT(l_, l_) += dyn_share.R_IMU(l_); //, l);
			}
        	Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse(); 
                                    
            Matrix<scalar_type, n, 1> dx_ = K * z; 

            P_ -= K * HP;
			x_.boxplus(dx_);
		}
		return;
	}
	
	void change_x(state &input_state)
	{
		x_ = input_state;

		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size())&&(!x_.SEN_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
	cov P_;
	state x_;
	// --- degeneracy-aware update (paper feature A) ---
	// Set from yaml after construction. When degen_en_ is true, a decaying accumulation of the
	// LiDAR pose information matrix (H^T R^{-1} H over the 6-DoF pose block) is maintained; its
	// weak eigen-directions are treated as degenerate and the Kalman gain along them is removed,
	// so those directions stay on the kinematic/IMU prediction. degen_en_ == false => no effect.
	bool degen_en_ = false;
	double degen_ratio_thr_ = 0.02;   // eigenvalue/lambda_max below this => degenerate direction
	double degen_decay_ = 0.95;       // exponential decay of the accumulated information matrix
	int degen_warmup_ = 50;           // updates to accumulate before suppression is allowed
	int degen_count_ = 0;
	Eigen::Matrix<scalar_type, 6, 6> degen_info_ = Eigen::Matrix<scalar_type, 6, 6>::Zero();
private:
	measurement m_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity();
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;
	processMatrix1 *f_x;
	processMatrix2 *f_w;

	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_dyn_share_modified_cov *h_dyn_share_modified_1;

	measurementModel_dyn_share_modified *h_dyn_share_modified_2;

	measurementModel_dyn_share_modified *h_dyn_share_modified_3;

	int maximum_iter = 0;
	scalar_type limit[n];
	
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
