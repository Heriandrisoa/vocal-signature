#include <sndfile.h>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main()
{
    SF_INFO sinfo;
    SNDFILE* file =
        sf_open("data/voice1.wav", SFM_READ, &sinfo);

    if (!file)
    {
        cerr << "cannot open file\n";
        return 1;
    }

    vector<float> audio(sinfo.frames);

    sf_read_float(
        file,
        audio.data(),
        sinfo.frames
    );

    sf_close(file);

    const int frame_size = 1024;
    const int hop_size   = 512;

    int sampleRate = sinfo.samplerate;

    double in[frame_size];
    fftw_complex out[frame_size / 2 + 1];
    fftw_plan plan =
        fftw_plan_dft_r2c_1d(
            frame_size,
            in,
            out,
            FFTW_ESTIMATE
        );

    cout << "Processing frames...\n";

    for (
        size_t start = 0;
        start + frame_size <= audio.size();
        start += hop_size
    )
    {
        for (int i = 0; i < frame_size; i++)
        {
            double hamming =
                0.54 -
                0.46 *
                cos(2.0 * M_PI * i / (frame_size - 1));

            in[i] =
                audio[start + i] * hamming;
        }

        // 2. FFT
        fftw_execute(plan);

        // 3. analyse spectre
        double max_mag = 0;
        int max_bin = 0;

        for (int k = 0; k < frame_size / 2 + 1; k++)
        {
            double re = out[k][0];
            double im = out[k][1];

            double mag =
                sqrt(re * re + im * im);

            if (mag > max_mag)
            {
                max_mag = mag;
                max_bin = k;
            }
        }

        double hz =
            max_bin *
            (double)sampleRate /
            frame_size;

        cout << "Frame start " << start
             << " | dominant freq = "
             << hz
             << " Hz"
             << " | mag = "
             << max_mag
             << endl;
    }

    fftw_destroy_plan(plan);

    return 0;
}