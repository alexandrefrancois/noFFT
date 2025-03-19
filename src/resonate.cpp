/*
MIT License

Copyright (c) 2025 Alexandre R. J. Francois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "./ResonatorBank.hpp"

namespace py = pybind11;

// #include <iostream>

namespace resonate
{
    py::array_t<float> resonate(py::array_t<float> input, float samplingRate, py::array_t<float> freqs, py::array_t<float> alphas, py::array_t<float> betas, int hopLength)
    {
        py::buffer_info buf = input.request(), fs = freqs.request(), as = alphas.request(), bs = betas.request();

        if (buf.ndim != 1 || fs.ndim != 1 || as.ndim != 1 || bs.ndim != 1)
            throw std::runtime_error("Number of dimensions must be one");

        if (fs.size != as.size)
            throw std::runtime_error("Frequencies and alphas must have same size");

        if (as.size != bs.size)
            throw std::runtime_error("Alphas and betas must have same size");

        size_t numSamples = buf.size;
        // std::cout << "Num samples: " << numSamples << std::endl;

        size_t numResonators = fs.size;
        // std::cout << "Num resonators: " << numResonators << std::endl;
        size_t twoNumResonators = 2 * numResonators;

        size_t numSlices = int(numSamples / hopLength);
        // std::cout << "Num slices: " << numSlices << std::endl;

        auto result = py::array_t<float>(numSlices * twoNumResonators); // we are returning complex values
        py::buffer_info rs = result.request();

        const float *frequenciesPtr = static_cast<float *>(fs.ptr);
        const float *alphasPtr = static_cast<float *>(as.ptr);
        const float *betasPtr = static_cast<float *>(bs.ptr);
        float *resultPtr = static_cast<float *>(rs.ptr);

        const float *inputPtr = static_cast<float *>(buf.ptr);

        ResonatorBank bank(numResonators, frequenciesPtr, alphasPtr, betasPtr, samplingRate);

        for (size_t idx = 0; idx < numSlices; idx++)
        {
            size_t inputOffset = idx * hopLength;
            bank.update(inputPtr + inputOffset, hopLength, 1);
            size_t resultOffset = idx * twoNumResonators;
            bank.getComplex(resultPtr + resultOffset, twoNumResonators);
        }

        // shape of results:
        // each slice of size twoNumResonators has
        // numResonators Re values and numResonators Im values (split complex)

        return result;
    }
}

PYBIND11_MODULE(noFFT, m)
{
    m.def("resonate", &resonate::resonate, "resonate");
}
