// --author-----------------------------------------------------------------------
// 
// Justin Asselin (juste-injuste)
// justin.asselin@usherbrooke.ca
// https://github.com/juste-injuste/Pinakas
// 
// --liscence---------------------------------------------------------------------
// 
// MIT License
// 
// Copyright (c) 2023 Justin Asselin (juste-injuste)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//  
// --versions---------------------------------------------------------------------
// version 1.0.0
// --inclusion guard--------------------------------------------------------------
#ifndef CHRONOMETRO_HPP
#define CHRONOMETRO_HPP
// --necessary standard libraries-------------------------------------------------
#include <chrono>
#include <iostream>
// --Chronometro library: backend forward declaration-----------------------------
namespace Chronometro { namespace Backend {
  enum Unit : char;
  class Stopwatch;
}}
// --Chronometro library: frontend forward declarations---------------------------
namespace Chronometro { inline namespace Frontend {
  using Backend::Unit;
  using Backend::Stopwatch;
}}
// --Pinakas library: backend struct and class definitions------------------------
namespace Chronometro { namespace Backend {
  enum Unit : char {
    ns = 0, us, ms, s, min, h
  };

  class Stopwatch {
    public:
      Stopwatch(Unit unit = ms);
      ~Stopwatch();
      void start(void);
      void start(Unit unit);
      void stop(void);
      void stop(Unit unit);
    private:
      Unit unit_;
      std::chrono::system_clock::time_point start_;
  };
}}
// --Automata library: backend struct and class member definitions----------------
namespace Chronometro { namespace Backend {
  Stopwatch::Stopwatch(Unit unit)
    : // member initialization list
    unit_(unit),
    start_(std::chrono::high_resolution_clock::now())
  {}

  Stopwatch::~Stopwatch()
  {
    stop();
  }

  void Stopwatch::start()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void Stopwatch::start(Unit unit)
  {
    unit_ = unit;
    start_ = std::chrono::high_resolution_clock::now();
  }

  void Stopwatch::stop(Unit unit)
  {
    unit_ = unit;
    stop();
  }

  void Stopwatch::stop(void)
  {
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "time elapsed: ";
    switch(unit_) {
      case ns:
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start_).count() << "ns";
        return;
      case us:
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start_).count() << "us";
        return;
      case ms:
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_).count() << "ms";
        return;
      case s:
        std::cout << std::chrono::duration_cast<std::chrono::seconds>(stop - start_).count() << 's';
        return;
      case min:
        std::cout << std::chrono::duration_cast<std::chrono::minutes>(stop - start_).count() << "min";
        return;
      case h:
        std::cout << std::chrono::duration_cast<std::chrono::hours>(stop - start_).count() << 'h';
        return;
    }
  }
}}
#endif