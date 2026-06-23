#include <sndfile.h>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
int main() {
    SF_INFO sinfo;
    SNDFILE* file = sf_open("data/voice1.wav", SFM_READ, &sinfo);
    
    if (!file)
    {
        cout << " errror opening file !" << endl;
        return -1;
    }
    vector<float> audio(sinfo.frames);
    sf_read_float(file, audio.data(), sinfo.frames);

    vector<string> repr;
    int pt_max = 0;
    int index_max = 0;
    for(int i = audio.size()/2; i < audio.size()/2 + 100; i++)
    {
        int pt = abs(audio[i]) * 1000;
        string cur;
        cout << audio[i];
        for (int j = 0; j < pt; j++)
            cur += "-";
        
        if (pt > pt_max)
        {
            pt_max = pt;
            index_max = i;
        }
        cout << cur << endl;
        repr.push_back(cur);
    }
    cout << endl;

    cout << "pt max: " << pt_max << endl;
        
    for(int i = pt_max - 1; i >= 0; i--)
    {
        for(int j = 0; j < repr.size(); j++)
        {
            if (repr[j].size() <= i)
                cout << "  ";
            else
                cout <<repr[j][i] << " ";
        }
        cout << endl;
    }
    sf_close(file);
    return 0;
}