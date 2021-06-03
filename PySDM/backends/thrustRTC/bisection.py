BISECTION = '''
    struct Bisect {
        static __device__ real_type bisect(
            real_type (*f)(real_type , void*), 
            void* args,
            real_type min, real_type max, 
            real_type fmin, real_type fmax,
            real_type tol
        ) {
            assert(min < max); // assumes in order
    
            if (fmin == 0) {
                return min;
            }
            if (fmax == 0) {
                return max;
            }
    
            // if bisection is ill-posed, return mid of the range
            if (fmin * fmax >= 0) {
                return (min + max) / 2; 
            }

            real_type mid = (min + max) / 2; 
            while (abs(max - min) > tol * abs(mid)) {
                if (mid == max || mid == min) {
                    break;
                }
    
                real_type fmid = f(mid, args);
    
                if (fmid == 0) {
                    break;
                }
                else if (fmid * fmin > 0) {
                    min = mid;
                    fmin = fmid;
                }
                else { 
                    max = mid; 
                    fmax = fmid;
                }
                mid = (min + max) / 2; 
            }
            return mid;
        }
    };
'''