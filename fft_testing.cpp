#include <sndfile.h>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <onnxruntime/onnxruntime_cxx_api.h>

using namespace std;

double hzToMel(double hz)
{
    return 2595.0 *
           log10(1.0 + hz / 700.0);
}

double melToHz(double mel)
{
    return 700.0 *
           (pow(10.0, mel / 2595.0) - 1.0);
}

double cosineSimilarity(
    const vector<float>& a,
    const vector<float>& b)
{
    double dot = 0;
    double na = 0;
    double nb = 0;

    for(size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    return dot /
           (sqrt(na) * sqrt(nb));
}

double euclideanDistance(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    double sum = 0.0;

    for(size_t i = 0; i < a.size(); i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

vector<float> getEmbeding(string path) {
SF_INFO sinfo;
    SNDFILE* file =
        sf_open(path.c_str(), SFM_READ, &sinfo);

    if (!file)
    {
        cerr << "cannot open file\n";
        return vector<float>();
    }
    const int n_mels = 80;
    vector<float> audio(sinfo.frames);
    vector<int64_t> shape =
    {
        1,
        300,
        80
    };
    sf_read_float(
        file,
        audio.data(),
        sinfo.frames
    );

    sf_close(file);

    constexpr int frame_size = 400;
    constexpr int hop_size   = 160;

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
    
    vector<float> features;
    cout << "Processing frames...\n";

    // processing by frame
    for (
        size_t start = 0;
        start + frame_size <= audio.size();
        start += hop_size
    )
    {
        // Hamming processing
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
        vector<double> power(frame_size / 2 + 1);

        for (int k = 0; k < frame_size / 2 + 1; k++)
        {
            double re = out[k][0];
            double im = out[k][1];

            power[k] = re * re + im * im;
        }

        // Mel calcul
        double melMin = hzToMel(0);
        double melMax =
            hzToMel(sampleRate / 2);

        vector<double> melPoints(n_mels + 2);

        for (int i = 0; i < n_mels + 2; i++)
        {
            melPoints[i] =
                melMin +
                (melMax - melMin) *
                i /
                (n_mels + 1);
        }

        // convert to Hz
        vector<double> hzPoints(n_mels + 2);

        for (int i = 0; i < n_mels + 2; i++)
        {
            hzPoints[i] =
                melToHz(melPoints[i]);
        }

        // convert to FFT
        vector<int> bins(n_mels + 2);

        for (int i = 0; i < n_mels + 2; i++)
        {
            bins[i] =
                floor(
                    (frame_size + 1)
                    * hzPoints[i]
                    / sampleRate
                );
        }

        // Mel extract energy
        vector<float> melEnergies(n_mels, 0);

                for (int m = 1; m <= n_mels; m++)
        {
            int left   = bins[m - 1];
            int center = bins[m];
            int right  = bins[m + 1];

            double energy = 0.0;

            for (int k = left; k < center; k++)
            {
                double weight =
                    (double)(k - left) /
                    (center - left);

                energy +=
                    power[k] * weight;
            }

            for (int k = center; k < right; k++)
            {
                double weight =
                    (double)(right - k) /
                    (right - center);

                energy +=
                    power[k] * weight;
            }

            melEnergies[m - 1] = energy;
        }

        // loging mel energy
        for (auto& v : melEnergies)
        {
            v = log(v + 1e-9);
        }
        
        features.insert(
            features.end(),
            melEnergies.begin(),
            melEnergies.end()
        );
 
    }


    // normalisation 
    double mean = 0;

    for (auto v : features)
    {
        mean += v;
    }

    mean /= features.size();
    double stddev = 0;

    for (auto v : features)
    {
        stddev += (v - mean) * (v - mean);
    }

    stddev = sqrt(stddev / features.size());
    for (auto& v : features)
    {
        v = (v - mean) / (stddev + 1e-9);
    }
    fftw_destroy_plan(plan);

    // normalizing data
    if (features.size() > 300* 80)
    {
        cout << "audio trop long, decoupage" << endl;
        features.resize(300*80);
    }
    else if (features.size() < 300*80)
    {
        cout << "audio trop court, completion " << endl;
        features.resize(300*80, 0.0f);
    }
    // setting up onnx 
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "speaker"
    );

    Ort::SessionOptions opts;

    Ort::Session session(
        env,
        "model.onnx",
        opts
    );

    Ort::MemoryInfo mem =
    Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );


    Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(
        mem,
        features.data(),
        features.size(),
        shape.data(),
        shape.size()
    );

    // infering
    const char* inputNames[] =
    {
        "features"
    };

    const char* outputNames[] =
    {
        "embedding"
    };

    auto outputs =
    session.Run(
        Ort::RunOptions{nullptr},
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    float* embedding =
    outputs[0]
        .GetTensorMutableData<float>();

    auto shapeOutput =
        outputs[0]
        .GetTensorTypeAndShapeInfo()
        .GetShape();

        for(auto s : shapeOutput)
            cout << s << " ";
    vector<float> emb(192);
    for (int i = 0; i < 192; i++)
    {
        emb[i] = embedding[i];
    }
    return emb;
}

int main()
{
    vector<float> a = getEmbeding("data/mahenina.wav");
    vector<float> b = getEmbeding("data/voice1.wav");
    cout << "cosine similariy :" << cosineSimilarity(a, b) << endl;
    cout << "Euclidian distance :" << euclideanDistance(a, b) << endl;

    // marge is upper than 99%
}