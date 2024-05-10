import numpy as np

Z = np.linspace(0.0, 3000.0, 20) / 1000
NC = np.array(
    [
        [
            2325.942800455995,
            2804.502235646257,
            3452.211545804375,
            4361.877640877965,
            5702.479217508372,
            7812.475748456826,
            11468.941392117338,
            18884.548901117727,
            39528.40155165309,
            235678.7814538211,
            9.90222080639631e6,
            9.944680372872213e6,
            9.948230520394495e6,
            9.92828984604994e6,
            9.49434950135968e6,
            60141.075852780945,
            23917.44136969187,
            13590.312401779991,
            8934.288272912649,
            6376.138306461998,
        ],
        [
            3698.0366744115295,
            5545.179144998437,
            10769.389564647368,
            27609.854578022492,
            80545.89895146045,
            226975.7205066433,
            557363.6510320791,
            1.120763299916621e6,
            1.7896000330774004e6,
            2.3017262507754182e6,
            2.5345248880887222e6,
            2.572187279672902e6,
            2.503338091749935e6,
            2.0706700213481337e6,
            1.0593380723943126e6,
            24531.98993962722,
            12175.909128446574,
            7113.883043905143,
            3778.6680192269737,
            1283.6531241529551,
        ],
        [
            29308.9239814404,
            62951.073404623625,
            125636.86235758603,
            224540.1168994546,
            353441.96241412446,
            492218.99897166644,
            622343.9910554249,
            742542.2675961609,
            864177.9086966328,
            990366.8478863802,
            1.09558521643041e6,
            1.1158491143098844e6,
            964952.628207187,
            616097.4232379115,
            219119.02162367874,
            11371.949314403208,
            5895.3752315160955,
            3015.304060595528,
            1232.292354232872,
            292.4413130466608,
        ],
        [
            97521.27310162572,
            129172.53896538781,
            169620.21868955236,
            220508.1361615754,
            282145.13792610687,
            351810.48781079333,
            421949.77587774675,
            479226.14530290593,
            505802.2781946063,
            484504.3914415042,
            408542.6017980009,
            291446.1011626797,
            166557.81121660033,
            70028.62549034489,
            18400.449377269604,
            2473.660072548018,
            1139.0577276446124,
            459.9643481033018,
            142.22834177371044,
            25.878852036127473,
        ],
    ]
)
QC = np.array(
    [
        [
            2.3259428004559947e-7,
            2.804502235646257e-7,
            3.452211545804375e-7,
            4.361877640877965e-7,
            5.702479217508372e-7,
            7.812475748456827e-7,
            1.1468941392117339e-6,
            1.8884548901117728e-6,
            3.952840155165309e-6,
            2.356787814538211e-5,
            0.0009902220806396308,
            0.0009944680372872211,
            0.0009948230520394495,
            0.000992828984604994,
            0.0009494349501359681,
            6.014107585278095e-6,
            2.3917441369691873e-6,
            1.3590312401779992e-6,
            8.934288272912649e-7,
            6.376138306461999e-7,
        ],
        [
            4.265806818800079e-7,
            7.226959794540242e-7,
            1.6428064435379529e-6,
            4.615558914198795e-6,
            1.348612628178059e-5,
            3.6201723706357274e-5,
            8.303537808232973e-5,
            0.00015533058013028498,
            0.0002320201174984399,
            0.0002829109283963567,
            0.0003012646493725666,
            0.0002994406742641612,
            0.00027764373421516376,
            0.00020956914385498947,
            9.25617594068194e-5,
            2.1258995452497305e-6,
            1.0810359738209302e-6,
            6.206079249799725e-7,
            3.1067233946395153e-7,
            9.491081416494235e-8,
        ],
        [
            5.618352425372331e-6,
            1.1770218400227012e-5,
            2.2366288063388795e-5,
            3.762110709028486e-5,
            5.54998275303101e-5,
            7.250222652832265e-5,
            8.641844054823889e-5,
            9.798474644403037e-5,
            0.00010924079793690472,
            0.00012017660480894775,
            0.00012629068345246645,
            0.00011904766069210393,
            9.162268485010753e-5,
            4.972230777150403e-5,
            1.4395817162693223e-5,
            8.521337702073747e-7,
            4.369433501348079e-7,
            2.0769518728226592e-7,
            7.508549422520445e-8,
            1.4963007624572663e-8,
        ],
        [
            1.6260372167235703e-5,
            2.0490225267114223e-5,
            2.5651591798270408e-5,
            3.1829228588792924e-5,
            3.882768326167003e-5,
            4.5950222670638464e-5,
            5.1859729650390664e-5,
            5.4700113591428155e-5,
            5.2641452465826274e-5,
            4.486890861872125e-5,
            3.2636650805065814e-5,
            1.9348899957061552e-5,
            8.816034080126421e-6,
            2.8502395829139177e-6,
            6.047936175098373e-7,
            1.1566416452964713e-7,
            4.709162101887405e-8,
            1.5623297187045765e-8,
            3.6750701849568283e-9,
            4.559677300872325e-10,
        ],
    ]
)
NR = np.array(
    [
        [
            0.00023259428004559948,
            0.0002804502235646257,
            0.0003452211545804375,
            0.0004361877640877965,
            0.0005702479217508372,
            0.0007812475748456827,
            0.0011468941392117338,
            0.0018884548901117728,
            0.003952840155165309,
            0.02356787814538211,
            0.9902220806396309,
            0.9944680372872212,
            0.9948230520394494,
            0.992828984604994,
            0.9494349501359681,
            0.0060141075852780945,
            0.002391744136969187,
            0.0013590312401779991,
            0.0008934288272912649,
            0.0006376138306461998,
        ],
        [
            55.0355816792709,
            249.68528047180155,
            1034.3817765007677,
            3866.077247070702,
            12851.814845734823,
            37279.94364248224,
            91893.56711734981,
            185999.34732983317,
            299161.8790322643,
            379350.320678476,
            392478.950603419,
            347895.16491086554,
            262449.6369779219,
            141679.59718938553,
            37206.72084218661,
            6.8802507875865935,
            1.7136513868814653,
            0.5874165391940699,
            0.17208977496020028,
            0.026697814223896093,
        ],
        [
            13655.554610747438,
            27911.62787282717,
            50923.87187335398,
            82248.95886378274,
            117360.18060553771,
            149358.1714808766,
            173573.8268549382,
            189858.33300005159,
            198836.00924994508,
            195824.0030565023,
            171321.0242639703,
            123013.33555976547,
            65636.70516287262,
            22185.06386129119,
            3451.462311112076,
            2.0858027698976924,
            0.5694690846699856,
            0.15231538567815248,
            0.029240832039356925,
            0.002809107337945199,
        ],
        [
            37856.926747433536,
            44438.051116661205,
            52668.90095026373,
            62795.34493974483,
            74622.14697527396,
            86823.38623152864,
            95861.49575136899,
            95476.05759287048,
            79830.57404431111,
            51788.5852015471,
            24745.71906550424,
            8601.6990557079,
            2134.676925640851,
            346.0192629372826,
            28.03307411508453,
            0.0833546554998565,
            0.018155077951843214,
            0.0032413168848044022,
            0.00040173684735678736,
            2.5076696448669475e-5,
        ],
    ]
)
QR = np.array(
    [
        [
            2.325942800455995e-13,
            2.804502235646257e-13,
            3.452211545804375e-13,
            4.361877640877965e-13,
            5.702479217508372e-13,
            7.812475748456827e-13,
            1.1468941392117339e-12,
            1.8884548901117728e-12,
            3.9528401551653096e-12,
            2.356787814538211e-11,
            9.90222080639631e-10,
            9.944680372872212e-10,
            9.948230520394494e-10,
            9.92828984604994e-10,
            9.494349501359683e-10,
            6.014107585278095e-12,
            2.391744136969187e-12,
            1.3590312401779993e-12,
            8.93428827291265e-13,
            6.376138306461999e-13,
        ],
        [
            5.3070243450859676e-8,
            2.463634950901659e-7,
            1.0490531657691007e-6,
            4.056100749708039e-6,
            1.4067577640555516e-5,
            4.304268677051551e-5,
            0.00011338196690501957,
            0.0002481999631916455,
            0.00043196677290087025,
            0.0005751254260669398,
            0.0005828928683903838,
            0.0004707060784035762,
            0.0003113131094598012,
            0.00014300068667404562,
            3.147028187828119e-5,
            4.336648200056951e-9,
            1.0808498301027295e-9,
            3.698118288127002e-10,
            1.0789166005028712e-10,
            1.6639453441939642e-11,
        ],
        [
            3.2862651183081475e-5,
            7.24365371893176e-5,
            0.0001429730191492526,
            0.00024940665557201744,
            0.00037981720050470844,
            0.0005007545865088965,
            0.0005710436749178275,
            0.0005676989612771941,
            0.0004983583023737646,
            0.00038937700308448665,
            0.00026771911074236024,
            0.00015479208657297362,
            6.864249934327006e-5,
            1.991561671453986e-5,
            2.7337234669538306e-6,
            1.2972853986561816e-9,
            3.534755221992608e-10,
            9.408068795803758e-11,
            1.7940744344265586e-11,
            1.7097382088747494e-12,
        ],
        [
            0.00033796259140966275,
            0.0002984254256737562,
            0.00025646452941255023,
            0.00021476545995241536,
            0.0001748360816014869,
            0.0001374450415495815,
            0.00010307828065789707,
            7.233288016165422e-5,
            4.616057763859581e-5,
            2.5769659505524617e-5,
            1.1985334389740117e-5,
            4.380953064857479e-6,
            1.1691073172814647e-6,
            2.0262209204983427e-7,
            1.7294198467455043e-8,
            4.954455149972736e-11,
            1.0663871343784473e-11,
            1.865553837839756e-12,
            2.2527694642843612e-13,
            1.385082972258734e-14,
        ],
    ]
)
