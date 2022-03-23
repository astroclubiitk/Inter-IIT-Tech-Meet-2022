# JuX : Solar X-Ray Burst Detector

This package was created by Team 10 on the account of Inter IIT Techmeet 10.
It's concerned with identification and analysis of solar xray bursts in provided light curves.

How to install it?  
While being in the `jux` package directory, do

```bash
pip install .
```

Dependencies will build and you can proceed using it.

This is the package structure:

```
.
├── examples
│   ├── ch2_xsm_20211111_v1_level2.lc
│   ├── isf.pickle
│   ├── script_outliers.py
│   └── script.py
├── install.sh
├── jux
│   ├── create_df_minmax.py
│   ├── denoise.py
│   ├── false_positive_detection.py
│   ├── file_handler.py
│   ├── flare_detect_minmax.py
│   ├── flare_detect_thresh.py
│   ├── helper.py
│   ├── __init__.py
│   ├── jux.py
│   ├── params.py
│   └── version.py
├── LICENSE
├── MANIFEST.in
├── README.md
└── setup.py
```

The main package lies in `./jux`.

- `./jux/helper.py` : Contains helper functions including analytical functions that were fitted to the posterior part of a flare.
- `./jux/denoise.py` : Contains 3 algorithms including smoothening with fft, smoothening with median windowing and moving average smoothening.
- `./jux/flare_detect_minmax.py` : Contains 7 deterministic and condition based filters that help pick out bursts from de-noised data.
- `./jux/jux.py` : The main file containing class `Lightcurve` that accepts file path as argument, The main function will execute all de-noising and filtering and give out flare details and model fitted parameters that can then be used in the Isolation Classifier to detect false positives.
- `./jux/false_positive_detection.py` : Contains the functions that can be uses sklearn's Isolation Classifier to pick out outliers in the flares detected by the algorithm implemented above.
- `./jux/params.py` : Contains parameters that control all of the algorithms, context mentioned in the file itself.
- `./jux/create_df_minmax.py` : Contains functions that handles the dataframe given as the output by filtering.

## Algorithms explained

### Smoothening of LC

Lightcurve we trained upon has quite a bit of noise, so for de-noising we tried the following:

- Moving Averaging : Taking a small window and taking its mean as the corresponding value for the whole interval, thus effectively reducing the time scale thus this is followed by interpolation
- Interpolation :
  - Linear : We opted for Linear Interpolation because of its faster time complexity and not much effect in the results to our filtering algorithms
  - Spline : Spline interpolation was tried and performed the best with the filters, but took much longer time, with complexity O(m^2\*n)
- FFT Smoothing : Taking the FFT of the whole Light Curve wouldn't be possible as the signal is discontinuous, because we are only considering the GTI (Good Time Intervals). Thus doing that for every interval and again linear interpolation was done.

### Filters used

- Filter 0 : Minima and Maxima detected
- Filter 1 : Pairing of minima and maxima
- Filter 2 : Slope Thresholding implemented
- Filter 3 : Height Thresholding implemented
- Filter 4 : Repeated pairs filtered off
- Filter 5 : Close flares filtered off

These filters together actually gave a great result in picking out what we perceived as flares, sharp rises and slow decays.

## Examples

The package contains 2 examples to run the algorithm. Check out `./examples/script.py`. To run it:

```bash
python ./script.py ./ch2_xsm_20211111_v1_level2.lc
```

This would run the script with this LC Curve as the input signal, you would get the dataframe with features as the output on the terminal itself.

The second script `./examples/script_outliers.py` runs the outlier detector algorithm with premade pickle object of an Isolation Forest. To run it:

```bash
python ./script_outliers.py ./ch2_xsm_20211111_v1_level2.lc ./isf.pickle
```

This would add an extra column of if the algorithm thinks that particular flare is an outlier or not.

### Customize

To customize and tweak the algorithm to perform better, you can play around with the values in `./jux/params.py`, though these were the optimum values found after playing around and some paper references.

### Training

Run the algorithm for multiple LC files and make a pickle for yourself with `./jux/false_positive_detection.py`. Check out its function of `train_isolationForest_`. Make a better pickle, else I will try to update this.
