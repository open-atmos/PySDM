"""
Created at 10.12.2019
"""

from functools import reduce
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from ..conf import trtc
from .precision_resolver import PrecisionResolver


class AlgorithmicMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        dim = trtc.DVInt64(dim)
        idx_length = trtc.DVInt64(position_in_cell.shape[1])
        courant_length = trtc.DVInt64(courant.shape[0])
        loop = trtc.For(
            ['dim', 'idx_length', 'displacement', 'courant', 'courant_length', 'cell_origin', 'position_in_cell'], "i",
            f'''
            // Arakawa-C grid
            int _l_0 = cell_origin[i + 0];
            int _l_1 = cell_origin[i + idx_length];
            int _l = _l_0 + _l_1 * courant_length;
            int _r_0 = cell_origin[i + 0] + 1 * (dim == 0);
            int _r_1 = cell_origin[i + idx_length] + 1 * (dim == 1);
            int _r = _r_0 + _r_1 * courant_length;
            int omega = position_in_cell[i + idx_length * dim];
            int c_r = courant[_r];
            int c_l = courant[_l];
            displacement[i, dim] = {scheme(None, None, None)}
            ''')
        loop.launch_n(
            displacement.shape[1],
            [dim, idx_length, displacement.data, courant.data, courant_length, cell_origin.data, position_in_cell.data])

    __coalescence_body = trtc.For(
        ['n', 'volume', 'idx', 'idx_length', 'intensive', 'intensive_length', 'extensive', 'extensive_length', 'gamma',
         'healthy', 'adaptive', 'subs', 'adaptive_memory'], "i", '''
        if (gamma[i] == 0) {
            adaptive_memory[i] = 1;
            return;
        }

        int j = idx[i];
        int k = idx[i + 1];

        if (n[j] < n[k]) {
            j = idx[i + 1];
            k = idx[i];
        }
        int g = (int)(n[j] / n[k]);
        if (adaptive) {
            adaptive_memory[i] = (int)(gamma[i] * subs / g);
        }
        if (g > gamma[i]) {
            g = gamma[i];
        }
        if (g == 0) {
            return;
        }
            
        int new_n = n[j] - g * n[k];
        if (new_n > 0) {
            n[j] = new_n;
            
            for (int attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + k] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
            }
            for (int attr = 0; attr < extensive_length; attr+=idx_length) {
                extensive[attr + k] += g * extensive[attr + j];
            }
        }
        else {  // new_n == 0
            n[j] = (int)(n[k] / 2);
            n[k] = n[k] - n[j];
            for (int attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + j] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
                intensive[attr + k] = intensive[attr + j];
            }
            for (int attr = 0; attr < extensive_length; attr+=idx_length) {
                extensive[attr + j] = g * extensive[attr + j] + extensive[attr + k];
                extensive[attr + k] = extensive[attr + j];
            }
        }
        if (n[k] == 0 || n[j] == 0) {
            healthy[0] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy, adaptive, subs, adaptive_memory):
        idx_length = trtc.DVInt64(len(idx))
        intensive_length = trtc.DVInt64(len(intensive))
        extensive_length = trtc.DVInt64(len(extensive))
        adaptive_device = trtc.DVBool(adaptive)
        subs_device = trtc.DVInt64(subs)
        AlgorithmicMethods.__coalescence_body.launch_n(length - 1,
                                                       [n.data, volume.data, idx.data, idx_length, intensive.data,
                                                        intensive_length, extensive.data, extensive_length, gamma.data,
                                                        healthy.data, adaptive_device, subs_device,
                                                        adaptive_memory.data])
        return trtc.Reduce(adaptive_memory.data.range(0, length - 1), trtc.DVInt64(0), trtc.Maximum())

    __compute_gamma_body = trtc.For(['prob', 'rand'], "i", '''
        prob[i] = ceil(prob[i] - rand[(int)(i / 2)]);
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def compute_gamma(prob, rand):
        AlgorithmicMethods.__compute_gamma_body.launch_n(len(prob), [prob.data, rand.data])

    __radius_lambda = """
        auto radius = [](double f01_volume){
            return pow( (f01_volume * 3 /4 / 3.1415926535897932384626) , (1/3));
        };
    """

    __volume_lambda = """
        auto volume = [](double f02_x){
            return exp(f02_x);
        };
    """

    __x_lambda = """
        auto x = [](double f03_volume){
            return log(f03_volume);
        };
    """

    __dx_dt_lambda = """
        ##radius_lambda##
        ##volume_lambda##
    
        auto dx_dt = [radius, volume] (double f04_x, double f04_dr_dt){
            auto r = radius(volume(f04_x));
            return 3/ r * f04_dr_dt;
        };
    """.replace("##radius_lambda##", __radius_lambda)\
       .replace("##volume_lambda##", __volume_lambda)


    __dr_dt_MM_lambda = """ // CALOSC WYZBYTA CONSTOW, MAJOR ISSUE
        auto dr_dt_MM = [](double f05_r, double f05_T, double f05_p, double f05_RH, double f05_kp, double f05_rd){
            
            auto RH_eq = [](double f06_r, double f06_T, double f06_kp, double f06_rd){
                auto B = [](double f07_kp, double f07_rd){
                    return 1;
                };
                auto A = [](double f08_T){
                    return 1;
                };
            
                return 1;
            };
            
            auto Fd = [](double f09_T, double f09_DResult){
                return 1;
            };
            
            auto Fk = [](double f10_T, double f10_K, double f10_lv){
                return 1;
            };
            
            auto lv = [](double f11_T)
            {
                return 1;
            };
            
            auto K = [](double f05_1_r, double f05_1_T, double f05_1_p)
            {
                return 1;
            };
            
            auto D = [](double f05_2_r, double f05_2_T)
            {
                return 1;
            };
            
            auto nom = (f05_RH - RH_eq(f05_r, f05_T, f05_kp, f05_rd));
            auto den =   Fd(f05_T, D(f05_r, f05_T)) +Fk(f05_T, K(f05_r, f05_T, f05_p), lv(f05_T));
            
            return 1 / f05_r * nom / den;
        };
    """

    __minfun_lambda = """
        auto minfun = [volume,radius,dr_dt_MM, dx_dt](double f12_x_new, double f12_x_old, double f12_dt, double f12_T, double f12_p, double f12_RH, double f12_kappa, double f12_rd){
            auto r_new = radius(volume(f12_x_new));
            auto dr_dt = dr_dt_MM(r_new, f12_T,f12_p,f12_RH,f12_kappa,f12_rd);
            return f12_x_old - f12_x_new + f12_dt * dx_dt(f12_x_new, dr_dt);
        };
        """

    __bisec_lambda = """
        auto bisec = [volume, radius, dx_dt] (
        double f_13_a,
        double f_13_interval,
        double f_13_x_old,
        double f_13_dt,
        double f_13_T,
        double f_13_p, 
        double f_13_RH,
        double f_13_kappa,
        double f_13_rd,
        double f_13_rtol){
            auto b = f_13_a + f_13_interval;
        
            ##dr_dt_MM_lambda##
            ##min_fun##
        
            auto fa = minfun(f_13_a, f_13_x_old, f_13_dt, f_13_T, f_13_p, f_13_RH, f_13_kappa, f_13_rd);
            auto fb = minfun(b, f_13_x_old, f_13_dt, f_13_T, f_13_p, f_13_RH, f_13_kappa, f_13_rd);
            
            auto counter = 0;
            while ( fa * fb > 0){
                counter++;
                if (counter > 100)
                    auto z = 1/0; // wyjatek
                
                b = f_13_a + f_13_interval * pow((double)2, counter);
                fb = minfun(b, f_13_x_old, f_13_dt, f_13_T, f_13_p, f_13_RH, f_13_kappa, f_13_rd);
            
            }
            if (b<f_13_a){
                auto tmp = b;
                b = f_13_a;
                f_13_a = tmp;
                
                tmp = fb;
                fb = fa;
                fa = tmp;
            }
            
            
            auto x_new = (f_13_a+b)/2;        
            while(1){
                if ( (b-f_13_a) < f_13_rtol * abs(x_new))
                    break;
            
                auto f_new = minfun(x_new, f_13_x_old, f_13_dt, f_13_T, f_13_p, f_13_RH, f_13_kappa, f_13_rd);
                
                if (f_new * fa > 0){
                    f_13_a = x_new;
                    fa= f_new;
                }else{
                    b = x_new;
                }
            }
            
            return x_new;
        };
    """.replace("##dr_dt_MM_lambda##", __dr_dt_MM_lambda)\
       .replace("##min_fun##", __minfun_lambda)


    __calculate_ml_old_lambda = """
        auto calculate_ml_old = []( 
        VectorView<double> f_14_v, 
        VectorView<int64_t> f_14_n,
        int f_14_cell_idx){
            auto result = 0;
            // TODO: to for po cell_idx, tutaj narazie przekazuje 1 !!!!     
            int drop = f_14_cell_idx;
            result += f_14_n[drop] * f_14_v[drop] *  1 /* const.rho_w */;
             
            return result;
        };
    """

    __calculate_ml_new_lambda = """ // FUNKCJA BEZ OBSLUGI DROP TEMPERATURE
        auto calculate_ml_new = [&ml_new, &ripening](
        double f_15_dt,
        double f_15_T,
        double f_15_p,
        double f_15_RH,
        VectorView<double> f_15_v,
        VectorView<double> f_15_particle_T,
        VectorView<int64_t> f_15_n,
        VectorView<double> f_15_vdry,
        int f_15_cell_idx,
        double f_15_kappa,
        double f_15_qv,
        double f_15_rtol_x){
        
            double result = 0;
            auto growing = 0;
            auto decreasing = 0;
        
            //for na krople :) 
        
            int drop = f_15_cell_idx;
                { 
                    ##x_lambda##
                    ##dx_dt_lambda##
                
                    auto x_old = x(f_15_v[drop]);
                    auto r_old = radius(f_15_v[drop]);
                    auto rd = radius(f_15_vdry[drop]);
                
                    ##__dr_dt_MM_lambda##
                
                    auto dr_dt_old = dr_dt_MM(r_old,f_15_T,f_15_p,f_15_RH, f_15_kappa, rd);
                    auto dx_old = f_15_dt * dx_dt(x_old, dr_dt_old);
                    
                    if(dx_old <0){
                        if( dx_old < x(f_15_vdry[drop]) - x_old) //maximum
                            dx_old = x(f_15_vdry[drop]) - x_old;
                    }
                    
                    
                    auto a = x_old;
                    auto interval = dx_old;
                    
                    ##bisec_lambda##
                    
                    auto x_new = bisec(a, interval, x_old, f_15_dt, f_15_T, f_15_p, f_15_RH, f_15_kappa, rd, f_15_rtol_x);
                    auto v_new = volume(x_new);
                    
                    if ( (abs(v_new - f_15_v[drop]) / v_new ) > 0.5 ){
                        if (v_new - f_15_v[drop] > 0){
                            growing++;
                        }else{
                            decreasing++;
                        }
                    }
                    f_15_v[drop] = v_new;
                    result += f_15_n[drop] * v_new * /* const.rho_w */ 1;
                    
                    ml_new = result;
                    ripening = (growing > 0 && decreasing > 0) ? 1 : 0;
                }
        };
    
    """.replace("##bisec_lambda##", __bisec_lambda)\
       .replace("##x_lambda##", __x_lambda)\
       .replace("##radius_lambda##", __radius_lambda)\
       .replace("##dx_dt_lambda##", __dx_dt_lambda)\
       .replace("##__dr_dt_MM_lambda##", __dr_dt_MM_lambda)

    __temperature_pressure_RH_lambda = """
        auto temperature_pressure_RH = [&T,&p,&RH](
        double f16_rhod,
        double f16_thd,
        double f16_qv){
            auto exponent = 1; // const.Rd / const.c_pd;
            auto pd = pow( (double)((f16_rhod * 1 * f16_thd) / pow((double)1000 , exponent)), (double)(1 / (1 - exponent))); //ODPOWIEDNIE consty, wyrzucilem wewnetrzne komentarze bo nie dzialalo
            auto R = /* const.Rv */ 1 / (1 / f16_qv + 1) +  1 /* const.Rd */ / (1 + f16_qv);

            auto T = pow( f16_thd * (pd / 1000 /* const.p1000 */) , exponent);
            auto p = f16_rhod * (1 + f16_qv) * R * T;
            
            auto pvs = [] (double f_16_1_T){
                return 1; // const :/
            };
            
            auto RH = (p - pd) / pvs(T);
        };

    """


    __step_lambda = """
            auto step = [&qv_new, &thd_new, &ripening_flag](
            VectorView<double> f17_v,
            VectorView<double> f17_particle_T,
            VectorView<int64_t> f17_n,
            VectorView<double> f17_vdry,
            int64_t f17_cell_idx, // TO BEDZIE JAKAS TABLICA RACZEJ !
            double f17_kappa,
            double f17_thd,
            double f17_qv,
            double f17_dthd_dt,
            double f17_dqv_dt,
            double f17_m_d,
            double f17_rhod_mean,
            double f17_rtol_x,
            double f17_dt,
            int64_t f17_n_substeps)
            {
                f17_dt /= f17_n_substeps;
                
                ##Calculate_ml_old_Lambda_Declaration##
                
                auto ml_old = calculate_ml_old(f17_v, f17_n, f17_cell_idx);
                
                auto ripenings = 0;
                for(int t=0; t<f17_n_substeps;t++){
                    f17_thd += f17_dt * f17_dthd_dt / 2;
                    f17_qv += f17_dt * f17_dqv_dt / 2;
                    
                    double T;
                    double p;
                    double RH;
                    
                    ##Temperature_pressure_RH_Lambda_Declaration##
                    temperature_pressure_RH(f17_rhod_mean, f17_thd, f17_qv);
                    
                    
                    double ml_new = 0; 
                    int ripening = 0; 
                    ##calculate_ml_new_Lambda_Declaration##
                    calculate_ml_new(f17_dt, T, p, RH, f17_v, f17_particle_T, f17_n, f17_vdry, f17_cell_idx, f17_kappa, f17_qv, f17_rtol_x);
                    
                    auto dml_dt = (ml_new - ml_old) / f17_dt;
                    auto dqv_dt_corr = - dml_dt / f17_m_d;
                    
                    auto dthd_dt = [](double f21_rhod, double f21_thd, double f21_T, double f21_sqv_dt){
                        return 1; //TODO
                    };
                    
                    auto dthd_dt_corr = dthd_dt(f17_rhod_mean, f17_thd, T, dqv_dt_corr);
                    f17_thd += f17_dt * (f17_dthd_dt / 2 + dthd_dt_corr);
                    f17_qv += f17_dt * (f17_dqv_dt / 2 + dqv_dt_corr);
                    ml_old = ml_new;
                    ripenings += ripening;
                }
                
                qv_new = f17_qv;
                thd_new = f17_thd;
                ripening_flag = ripenings;
            };
    """.replace("##Temperature_pressure_RH_Lambda_Declaration##", __temperature_pressure_RH_lambda)\
       .replace("##calculate_ml_new_Lambda_Declaration##", __calculate_ml_new_lambda)\
       .replace("##Calculate_ml_old_Lambda_Declaration##", __calculate_ml_old_lambda)


    __solve_lambda = """
        auto solver = [&qv_new,&thd_new,&substeps_hint,&ripening_flag](
            VectorView<double> f18_v,
            VectorView<double> f18_particles_temperatures,
            VectorView<int64_t> f18_n,
            VectorView<double> f18_vdry,
            int64_t f18_idx, // TO BEDZIE JAKAS TABLICA RACZEJ !
            double f18_kappa,
            double f18_thd,
            double f18_qv,
            double f18_dthd_dt,
            double f18_dqv_dt,
            double f18_m_d,
            double f18_rhod_mean,
            double f18_rtol_x,
            double f18_rtol_thd,
            double f18_dt,
            int64_t f18_n_substeps
        ){
           qv_new = 0;
           thd_new = 0;
           substeps_hint = 0;
           ripening_flag = 0;
           
           ##Step_Lambda_Declaration##
           
           step(f18_v, f18_particles_temperatures, f18_n, f18_vdry, f18_idx, f18_kappa, f18_thd, f18_qv, f18_dthd_dt, f18_dqv_dt, f18_m_d, f18_rhod_mean, f18_rtol_x, f18_dt, f18_n_substeps);
           
           substeps_hint = f18_n_substeps;
        }
    """.replace('##Step_Lambda_Declaration##', __step_lambda)


    __condensation_main_body = '''
    
        //Thread id to będzie iterator od thrusta, n_threads trzeba bedzie podac
        for (int i = thread_id ; i < n_cell ; i+= n_threads){
            int cell_id = cell_order[i];

            int cell_start = cell_start_arg[cell_id];
            int cell_end = cell_start_arg[cell_id + 1];
            int n_sd_in_cell = cell_end - cell_start;
            if (n_sd_in_cell == 0)
               continue;

            double dthd_dt = (pthd[cell_id] - thd[cell_id]) / dt;
            double dqv_dt = (pqv[cell_id] - qv[cell_id]) / dt;
            double rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2;
            double md = rhod_mean * dv_mean;
            
            double qv_new;
            double thd_new;
            int substeps_hint;
            int ripening_flag;
            
            ##Solve_Lambda_Declaration## ;

            
            //HACK
            
            int idx_start_end = idx[cell_start];

            solver(
                v, particle_temperatures, n, vdry,
                idx_start_end, kappa, thd[cell_id], qv[cell_id], dthd_dt, dqv_dt, md, rhod_mean,
                rtol_x, rtol_thd, dt, substeps[cell_id]
            );

            substeps[cell_id] = substeps_hint;
            ripening_flags[cell_id] += ripening_flag;

            pqv[cell_id] = qv_new;
            pthd[cell_id] = thd_new;
        }
    '''.replace('##Solve_Lambda_Declaration##', __solve_lambda)


    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order, ripening_flags
    ):
        n_threads = 64*2000 ; #TODO Wykrywanie, ja daje z palca

        threads_as_trtc = trtc.DVInt64(n_threads)
        cell_as_trtc = trtc.DVInt64(n_cell)
        dv_as_trtc=trtc.DVDouble(dv)
        kappa_as_trtc = trtc.DVDouble(kappa)
        rtol_x_as_trtc = trtc.DVDouble(rtol_x)
        rtol_thd_as_trtc = trtc.DVDouble(rtol_thd)
        dt_as_trtc = trtc.DVDouble(dt)
        cell_order_as_trtc = trtc.device_vector_from_numpy(cell_order)

        trtc.For(['n_threads', 'n_cell', 'cell_start_arg', 'v', 'particle_temperatures',
                  'n', 'vdry', 'idx', 'rhod', 'thd', 'qv','dv_mean','prhod','pthd','pqv','kappa',
                  'rtol_x', 'rtol_thd', 'dt', 'substeps', 'cell_order','ripening_flags'],
                 "thread_id" , AlgorithmicMethods.__condensation_main_body).launch_n(n_threads,
                                                                  [threads_as_trtc,
                                                                   cell_as_trtc,
                                                                   cell_start_arg.data,
                                                                   v.data,
                                                                   particle_temperatures.data,
                                                                   n.data,
                                                                   vdry.data,
                                                                   idx.data,
                                                                   rhod.data,
                                                                   thd.data,
                                                                   qv.data,
                                                                   dv_as_trtc,
                                                                   prhod.data,
                                                                   pthd.data,
                                                                   pqv.data,
                                                                   kappa_as_trtc,
                                                                   rtol_x_as_trtc,
                                                                   rtol_thd_as_trtc,
                                                                   dt_as_trtc,
                                                                   substeps.data,
                                                                   cell_order_as_trtc,
                                                                   ripening_flags.data])



    __flag_precipitated_body = trtc.For(['idx', 'idx_length', 'n_dims', 'healthy', 'cell_origin', 'position_in_cell',
                                         'volume', 'n'], "i", '''
        if (cell_origin[idx_length * (n_dims-1) + i] == 0 && position_in_cell[idx_length * (n_dims-1) + i] < 0) {
            idx[i] = idx_length;
            healthy[0] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def flag_precipitated(cell_origin, position_in_cell, volume, n, idx, length, healthy):
        idx_length = trtc.DVInt64(len(idx))
        n_dims = trtc.DVInt64(len(cell_origin.shape))
        AlgorithmicMethods.__flag_precipitated_body.launch_n(
            length, [idx.data, idx_length, n_dims, healthy.data, cell_origin.data, position_in_cell.data,
                     volume.data, n.data])
        return 0  # TODO

    __linear_collection_efficiency_body = trtc.For(
        ['A', 'B', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'G3', 'Mf', 'Mg', 'output', 'radii',
         'is_first_in_pair', 'unit'], "i", '''
        output[i] = 0;
        if (is_first_in_pair[i]) {
            real_type r = radii[i] / unit;
            real_type r_s = radii[i + 1] / unit;
            real_type p = r_s / r;
            if (p != 0 && p != 1) {
                real_type G = pow((G1 / r), Mg) + G2 + G3 * r;
                real_type Gp = pow((1 - p), G);
                if (Gp != 0) {
                    real_type D = D1 / pow(r, D2);
                    real_type E = E1 / pow(r, E2);
                    real_type F = pow((F1 / r), Mf) + F2;
                    output[i] = A + B * p + D / pow(p, F) + E / Gp;
                    if (output[i] < 0) {
                        output[i] = 0;
                    }
                }
            }
        }
    '''.replace("real_type", PrecisionResolver.get_C_type()))

    @staticmethod
    def linear_collection_efficiency(params, output, radii, is_first_in_pair, unit):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        dA = PrecisionResolver.get_floating_point(A)
        dB = PrecisionResolver.get_floating_point(B)
        dD1 = PrecisionResolver.get_floating_point(D1)
        dD2 = PrecisionResolver.get_floating_point(D2)
        dE1 = PrecisionResolver.get_floating_point(E1)
        dE2 = PrecisionResolver.get_floating_point(E2)
        dF1 = PrecisionResolver.get_floating_point(F1)
        dF2 = PrecisionResolver.get_floating_point(F2)
        dG1 = PrecisionResolver.get_floating_point(G1)
        dG2 = PrecisionResolver.get_floating_point(G2)
        dG3 = PrecisionResolver.get_floating_point(G3)
        dMf = PrecisionResolver.get_floating_point(Mf)
        dMg = PrecisionResolver.get_floating_point(Mg)
        dunit = PrecisionResolver.get_floating_point(unit)
        AlgorithmicMethods.__linear_collection_efficiency_body.launch_n(len(is_first_in_pair) - 1,
                                                                        [dA, dB, dD1, dD2, dE1, dE2, dF1, dF2, dG1, dG2,
                                                                         dG3, dMf, dMg, output.data, radii.data,
                                                                         is_first_in_pair.data, dunit])

    __interpolation_body = trtc.For(['output', 'radius', 'factor', 'a', 'b'], 'i', '''
        int r_id = (int)(factor * radius[i]);
        auto r_rest = (factor * radius[i] - r_id) / factor;
        output[i] = a[r_id] + r_rest * b[r_id];
    ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def interpolation(output, radius, factor, b, c):
        factor_device = trtc.DVInt64(factor)
        AlgorithmicMethods.__interpolation_body.launch_n(len(radius),
                                                         [output.data, radius.data, factor_device, b.data, c.data])

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme=None):
        return AlgorithmicMethods._sort_by_cell_id_and_update_cell_start

    __loop_reset = trtc.For(['vector_to_clean'], "i",
                            '''
                                vector_to_clean[i] = 0;
                            ''')

    __moments_body_0 = trtc.For(
        ['idx', 'min_x', 'attr', 'x_id', 'max_x', 'moment_0', 'cell_id', 'n', 'specs_idx_shape', 'moments',
         'specs_idx', 'specs_rank', 'attr_shape', 'moments_shape'], "fake_i",
        '''
            auto i = idx[fake_i];
            if (min_x < attr[attr_shape * x_id + i] && attr[attr_shape  * x_id + i] < max_x) {
                atomicAdd((unsigned long long int*)&moment_0[cell_id[i]], (unsigned long long int)n[i]);
                for (int k = 0; k < specs_idx_shape; k+=1) {
                    atomicAdd((real_type*) &moments[moments_shape * k + cell_id[i]], n[i] * pow((real_type)attr[attr_shape * specs_idx[k] + i], (real_type)specs_rank[k]));
                }
            }
        '''.replace("real_type", PrecisionResolver.get_C_type()))

    __moments_body_1 = trtc.For(['specs_idx_shape', 'moments', 'moment_0', 'moments_shape'], "c_id",
                                '''
                                    for (int k = 0; k < specs_idx_shape; k+=1) {
                                        if (moment_0[c_id] == 0) {
                                            moments[moments_shape * k  + c_id] = 0;
                                        } 
                                        else {
                                            moments[moments_shape * k + c_id] = moments[moments_shape * k + c_id] / moment_0[c_id];
                                        }
                                    }
                                ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        AlgorithmicMethods.__loop_reset.launch_n(moment_0.shape[0], [moment_0.data])
        AlgorithmicMethods.__loop_reset.launch_n(reduce(lambda x, y: x * y, moments.shape), [moments.data])

        AlgorithmicMethods.__moments_body_0.launch_n(length,
                                                     [idx.data, PrecisionResolver.get_floating_point(min_x), attr.data,
                                                      trtc.DVInt64(x_id),
                                                      PrecisionResolver.get_floating_point(max_x),
                                                      moment_0.data, cell_id.data, n.data,
                                                      trtc.DVInt64(specs_idx.shape[0]), moments.data,
                                                      specs_idx.data, specs_rank.data, trtc.DVInt64(attr.shape[1]),
                                                      trtc.DVInt64(moments.shape[1])])

        AlgorithmicMethods.__moments_body_1.launch_n(moment_0.shape[0],
                                                     [trtc.DVInt64(specs_idx.shape[0]), moments.data, moment_0.data,
                                                      trtc.DVInt64(moments.shape[1])])

    __normalize_body_0 = trtc.For(['cell_start', 'norm_factor', 'dt_div_dv'], "i", '''
        int sd_num = cell_start[i + 1] - cell_start[i];
        if (sd_num < 2) {
            norm_factor[i] = 0;
        }
        else {
            int half_sd_num = sd_num / 2;
            norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / half_sd_num;
        }
        ''')

    __normalize_body_1 = trtc.For(['prob', 'cell_id', 'norm_factor'], "i", '''
        prob[i] *= norm_factor[cell_id[i]];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        n_cell = cell_start.shape[0] - 1
        device_dt_div_dv = PrecisionResolver.get_floating_point(dt_div_dv)
        AlgorithmicMethods.__normalize_body_0.launch_n(n_cell, [cell_start.data, norm_factor.data, device_dt_div_dv])
        AlgorithmicMethods.__normalize_body_1.launch_n(prob.shape[0], [prob.data, cell_id.data, norm_factor.data])

    __remove_zeros_body = trtc.For(['data', 'idx', 'idx_length'], "i", '''
        if (idx[i] < idx_length && data[idx[i]] == 0) {
            idx[i] = idx_length;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def remove_zeros(data, idx, length) -> int:
        idx_length = trtc.DVInt64(idx.size())

        # Warning: (potential bug source): reading from outside of array
        AlgorithmicMethods.__remove_zeros_body.launch_n(length, [data, idx, idx_length])

        trtc.Sort(idx)

        result = idx.size() - trtc.Count(idx, idx_length)
        return result

    ___sort_by_cell_id_and_update_cell_start_body = trtc.For(['cell_id', 'cell_start', 'idx'], "i", '''
        if (i == 0) {
            cell_start[cell_id[idx[0]]] = 0;
        } 
        else {
            int cell_id_curr = cell_id[idx[i]];
            int cell_id_next = cell_id[idx[i + 1]];
            int diff = (cell_id_next - cell_id_curr);
            for (int j = 1; j < diff + 1; j += 1) {
                cell_start[cell_id_curr + j] = idx[i + 1];
            }
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def _sort_by_cell_id_and_update_cell_start(cell_id, cell_start, idx, length):
        # TODO !!!
        assert max(cell_id.to_ndarray()) == 0
        trtc.Fill(cell_start.data, trtc.DVInt64(length))
        AlgorithmicMethods.___sort_by_cell_id_and_update_cell_start_body.launch_n(length - 1,
                                                                                  [cell_id.data, cell_start.data,
                                                                                   idx.data])
        return idx
