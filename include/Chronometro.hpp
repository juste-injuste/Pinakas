/*---author-----------------------------------------------------------------------------------------

Justin Asselin (juste-injuste)
justin.asselin@usherbrooke.ca
https://github.com/juste-injuste/Chronometro

-----liscence---------------------------------------------------------------------------------------
 
MIT License

Copyright (c) 2023 Justin Asselin (juste-injuste)

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
 
-----versions---------------------------------------------------------------------------------------

Version 1.0.0 - Initial release

-----description------------------------------------------------------------------------------------

Chronometro is a simple and lightweight C++11 (and newer) library that allows you to measure the
execution time of functions or code blocks. See the included README.MD file for more information.

-----inclusion guard------------------------------------------------------------------------------*/
#ifndef CHRONOMETRO_HPP
#define CHRONOMETRO_HPP
// --necessary standard libraries-------------------------------------------------------------------
#include <chrono>   // for clocks and time representations
#include <iostream> // for std::cout, std::cerr
#include <cstddef>  // for size_t
#include <ostream>  // for std::ostream, std::endl
// --Chronometro library----------------------------------------------------------------------------
namespace Chronometro
{
// --Chronometro library: frontend forward declarations---------------------------------------------
  inline namespace Frontend
  {
    // library version
    #define CHRONOMETRO_VERSION       001000000L
    #define CHRONOMETRO_VERSION_MAJOR 1
    #define CHRONOMETRO_VERSION_MINOR 0
    #define CHRONOMETRO_VERSION_PATCH 0
    
    // bring clocks to frontend
    using std::chrono::system_clock;
    using std::chrono::steady_clock;
    using std::chrono::high_resolution_clock;

    // time units for displaying
    enum class Unit : unsigned char {
      ns,       // nanoseconds
      us,       // microseconds
      ms,       // milliseconds
      s,        // seconds
      min,      // minutes
      h,        // hours
      automatic // deduce automatically the appropriate unit
    };

    // measure elapsed time
    template<typename C = high_resolution_clock>
    class Stopwatch;

    // measure function execution time
    template<typename C = high_resolution_clock, typename F, typename... A>
    typename C::duration execution_time(const F function, const size_t repetitions, const A... arguments);

    // measure function execution time without function calling via pointers
    #define CHRONOMETRO_EXECUTION_TIME(function, repetitions, ...)

    // output stream
    std::ostream out_stream(std::cout.rdbuf());
    // error stream
    std::ostream err_stream(std::cerr.rdbuf());
    // warning stream
    std::ostream wrn_stream(std::cerr.rdbuf());
  }
// --Chronometro library: frontend struct and class definitions-------------------------------------
  inline namespace Frontend
  {
    template<typename C>
    class Stopwatch final {
      public:
        inline explicit Stopwatch(const Unit unit = Unit::automatic) noexcept;
        // start measuring time
        void start(void) noexcept;
        // pause time measurement
        typename C::duration pause(void) noexcept;
        // stop time measurement and display elapsed time
        typename C::duration stop(void) noexcept;
        // reset measured time and start measuring time
        inline void restart(void) noexcept;
        // set unit
        inline void set(const Unit unit) noexcept;
      private:
        // unit to be used when displaying elapsed time
        Unit unit_;
        // used to keep track stopwatch status
        bool paused_;
        // time either at construction or from last start/restart 
        typename C::time_point start_;
        // measured elapsed time
        typename C::duration duration_;
    };
  }
// --Chronometro library: backend forward declaration-----------------------------------------------
  namespace Backend
  {
    // returns the appropriate unit to display time
    Unit appropriate_unit(const std::chrono::nanoseconds::rep nanoseconds);
  }
// --Chronometro library: frontend definitions------------------------------------------------------
  inline namespace Frontend
  {
    template<typename C>
    Stopwatch<C>::Stopwatch(const Unit unit) noexcept
      : // member initialization list
      unit_(unit),
      paused_(false),
      start_(C::now()),
      duration_(0)
    {}

    template<typename C>
    void Stopwatch<C>::start(void) noexcept
    {
      if (paused_) {
        // unpause
        paused_ = false;

        // measure current time
        start_ = C::now();
      }
      else wrn_stream << "warning: Stopwatch: already started" << std::endl;
    }

    template<typename C>
    typename C::duration Stopwatch<C>::pause(void) noexcept
    {
      // measure elapsed time
      const typename C::duration duration = C::now() - start_;

      // add elapsed time up to now if not paused
      if (paused_)
        wrn_stream << "warning: Stopwatch: already paused" << std::endl;
      else {
        duration_ += duration;
        paused_ = true;
      }

      return duration_;
    }

    template<typename C>
    typename C::duration Stopwatch<C>::stop(void) noexcept
    {
      // pause time measurement
      pause();

      // measured time in nanoseconds
      const std::chrono::nanoseconds::rep nanoseconds = std::chrono::nanoseconds(duration_).count();

      // if unit_ == automatic, deduce the appropriate unit
      switch((unit_ == Unit::automatic) ? Backend::appropriate_unit(nanoseconds) : unit_) {
        case Unit::ns:
          out_stream << "elapsed time: " << nanoseconds << " ns" << std::endl;
          break;
        case Unit::us:
          out_stream << "elapsed time: " << nanoseconds / 1000 << " us" << std::endl;
          break;
        case Unit::ms:
          out_stream << "elapsed time: " << nanoseconds / 1000000 << " ms" << std::endl;
          break;
        case Unit::s:
          out_stream << "elapsed time: " << nanoseconds / 1000000000 << " s" << std::endl;
          break;
        case Unit::min:
          out_stream << "elapsed time: " << nanoseconds / 60000000000 << " min" << std::endl;
          break;
        case Unit::h:
          out_stream << "elapsed time: " << nanoseconds / 3600000000000 << " h" << std::endl;
          break;
        default: err_stream << "error: Stopwatch: invalid time unit" << std::endl;
      }

      return duration_;
    }

    template<typename C>
    void Stopwatch<C>::restart(void) noexcept
    {
      // reset measured duration
      duration_ = typename C::duration(0);

      // unpause
      paused_ = false;

      // measure current time
      start_ = C::now();
    }

    template<typename C>
    void Stopwatch<C>:: set(const Unit unit) noexcept
    {
      // validate unit
      if (unit > Unit::automatic) {
        wrn_stream << "warning: Stopwatch: invalid unit; automatic used instead" << std::endl;
        unit_ = Unit::automatic;
      }
      else unit_ = unit;
    }

    template<typename C, typename F, typename... A>
    typename C::duration execution_time(const F function, const size_t repetitions, const A... arguments)
    {
      Stopwatch<C> stopwatch(Unit::automatic);

      for (size_t iteration = 0; iteration < repetitions; ++iteration)
        function(arguments...);

      return stopwatch.stop();
    }

    #undef  CHRONOMETRO_EXECUTION_TIME
    #define CHRONOMETRO_EXECUTION_TIME(function, repetitions, ...)           \
      [&](void) {                                                            \
        Chronometro::Stopwatch<> stopw_atch;                                 \
        const size_t repet_itions = repetitions;                             \
        for (size_t itera_tion = 0; itera_tion < repet_itions; ++itera_tion) \
          function(__VA_ARGS__);                                             \
        return stopw_atch.stop();                                            \
      }()
  }
// --Chronometro library: backend definitions-------------------------------------------------------
  namespace Backend
  {
    Unit appropriate_unit(const std::chrono::nanoseconds::rep nanoseconds)
    {
      // 10 h < duration
      if (nanoseconds > 36000000000000)
        return Unit::h;

      // 10 min < duration <= 10 h
      if (nanoseconds > 600000000000)
        return Unit::min;

      // 10 s < duration <= 10 m
      if (nanoseconds > 10000000000)
        return Unit::s;

      // 10 ms < duration <= 10 s
      if (nanoseconds > 10000000)
        return Unit::ms;

      // 10 us < duration <= 10 ms
      if (nanoseconds > 10000)
        return Unit::us;

      // duration <= 10 us
      return Unit::ns;
    }
  }
}
#endif
