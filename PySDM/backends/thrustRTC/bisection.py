BISECTION='''
struct Bisect
{
    __device__
    static real_type bisect(
        real_type (*f)(const real_type&, void*), 
        void *args,
        real_type min, real_type max, 
        real_type fmin, real_type fmax,
        const real_type &tol
    )
    {
        assert(min < max); // assumes in order

        if (fmin == 0) return min;
        if (fmax == 0) return max;

        // if bisection is ill-posed, return mid of the range
        if (fmin * fmax >= 0) return (min + max) / 2; 

        real_type mid; 
        while (
            mid = (min + max) / 2, 
            abs(max - min) > tol
        )
        {
            if (mid == max || mid == min) break;

            real_type fmid = f(mid, args);

            if (fmid == 0) break;
            else if (fmid * fmin > 0) 
                min = mid, fmin = fmid;
            else 
                max = mid, fmax = fmid;
        }
        return mid;
    }
};
'''