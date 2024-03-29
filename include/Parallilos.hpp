/*---author-----------------------------------------------------------------------------------------

Justin Asselin (juste-injuste)
justin.asselin@usherbrooke.ca
https://github.com/juste-injuste/Parallilos

-----licence----------------------------------------------------------------------------------------
 
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

-----description------------------------------------------------------------------------------------

Parallilos is a simple and lightweight C++11 (and newer) library that abstracts away SIMD usage to
facilitate generic parallelism.

-----inclusion guard------------------------------------------------------------------------------*/
#ifndef PARALLILOS_HPP
#define PARALLILOS_HPP
// --necessary standard libraries-------------------------------------------------------------------
#include <cstddef>      // for size_t
#include <cmath>        // for std::sqrt
#include <cstdlib>      // for std::malloc, std::free
#include <ostream>      // for std::ostream
#include <iostream>     // for std::cerr
# include <type_traits> // for std::is_arithmetic
#if defined(PARALLILOS_WARNINGS)
# include <string>      // for std::string, std::to_string
# include <type_traits> // for std::is_floating_point, std::is_unsigned
# include <typeinfo>    // to use operator typeid
#endif
// --Parallilos library-----------------------------------------------------------------------------
#if defined(__GNUC__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
# if (__cplusplus < 201703L) and defined(_GLIBCXX_HAVE_ALIGNED_ALLOC)
#   define PARALLILOS_HAS_ALIGNED_ALLOC
# endif
#elif defined(__clang__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
# if (__cplusplus < 201703L) and defined(_LIBCPP_HAS_C11_FEATURES)
#   define PARALLILOS_HAS_ALIGNED_ALLOC
# endif
#elif defined(__MINGW32__) || defined(__MINGW64__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(__apple_build_version__)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_INLINE __forceinline
#elif defined(__INTEL_COMPILER)
# define PARALLILOS_COMPILER_SUPPORTS_SSE
# define PARALLILOS_COMPILER_SUPPORTS_AVX
# define PARALLILOS_COMPILER_SUPPORTS_SVML
# define PARALLILOS_INLINE __forceinline
#elif defined(__ARMCC_VERSION)
# define PARALLILOS_COMPILER_SUPPORTS_NEON
# define PARALLILOS_INLINE __forceinline
#else
# if __cplusplus >= 202302L
#   warning "warning: Parallilos: your compiler is not supported."
# endif
# define PARALLILOS_INLINE inline
#endif
#if defined(PARALLILOS_COMPILER_SUPPORTS_AVX)
# undef PARALLILOS_COMPILER_SUPPORTS_AVX
# if defined(__AVX512F__)
#   define PARALLILOS_AVX512F
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__AVX2__)
#   define PARALLILOS_AVX2
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__FMA__)
#   define PARALLILOS_FMA
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__AVX__)
#   define PARALLILOS_AVX
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
#endif

#if defined(PARALLILOS_COMPILER_SUPPORTS_SSE)
# undef PARALLILOS_COMPILER_SUPPORTS_SSE
# if defined(__SSE4_2__)
#   define PARALLILOS_SSE4_2
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__SSE4_1__)
#   define PARALLILOS_SSE4_1
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__SSSE3__)
#   define PARALLILOS_SSSE3
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__SSE3__)
#   define PARALLILOS_SSE3
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__SSE2__)
#   define PARALLILOS_SSE2
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
# if defined(__SSE__)
#   define PARALLILOS_SSE
#   define PARALLILOS_SIMD_HEADER <immintrin.h>
# endif
#endif

#if defined(PARALLILOS_COMPILER_SUPPORTS_NEON)
# undef PARALLILOS_COMPILER_SUPPORTS_NEON
# if defined(__ARM_NEON) or defined(__ARM_NEON__)
#   if defined(__ARM_ARCH_64)
#     define PARALLILOS_NEON64
#     define PARALLILOS_SIMD_HEADER <arm64_neon.h>
#   else
#     define PARALLILOS_NEON
#     define PARALLILOS_SIMD_HEADER <arm_neon.h>
#   endif
# endif
#endif

#if defined(PARALLILOS_SIMD_HEADER)
# if not defined(__OPTIMIZE__)
#   define __OPTIMIZE__
#   include PARALLILOS_SIMD_HEADER
#   undef  __OPTIMIZE__
# else
#   include PARALLILOS_SIMD_HEADER
# endif
#endif

#if defined(PARALLILOS_COMPILER_SUPPORTS_SVML)
# undef PARALLILOS_COMPILER_SUPPORTS_SVML
# define PARALLILOS_SVML
#endif


namespace Parallilos
{
  template<typename T>
  class SIMD;

  namespace Version
  {
    constexpr long NUMBER = 000001000;
    constexpr long MAJOR  = 000      ;
    constexpr long MINOR  =    001   ;
    constexpr long PATCH  =       000;
  }

  namespace Global
  {
    std::ostream wrn{std::cerr.rdbuf()};
  }

  namespace Backend
  {
# if defined(PARALLILOS_WARNINGS)
    template<typename T>
    inline std::string type_name()
    {
      if (std::is_floating_point<T>::value)
      {
        return "float" + std::to_string(sizeof(T) * 8);
      }

      if (std::is_unsigned<T>::value)
      {
        return "uint" + std::to_string(sizeof(T) * 8);
      }

      return "int" + std::to_string(sizeof(T) * 8);
    }

    template<typename T, typename... Tn>
    inline auto get_type_name() -> typename std::enable_if<sizeof...(Tn) == 0, std::string>::Type
    {
      return type_name<T>();
    }

    template<typename T, typename... Tn>
    inline auto get_type_name() -> typename std::enable_if<sizeof...(Tn) != 0, std::string>::Type
    {
      return type_name<T>() + ", " + get_type_name<Tn...>();
    }
#   define PARALLILOS_TYPE_WARNING(...)                                        \
      Global::wrn << "warning: Parallilos: " << __func__ << '(' <<             \
      Backend::get_type_name<__VA_ARGS__>() << "): SIMD not used" << std::endl
# else
#   define PARALLILOS_TYPE_WARNING(...) /* to enable warnings #define PARALLILOS_WARNINGS */
# endif

    template<size_t size>
    class Parallel
    {
    public:
      inline explicit Parallel(const size_t n_elements) noexcept :
        current_index{0},
        passes_left{size ? (n_elements / size) : 0}
      {}
      inline size_t operator*() noexcept               {return current_index;}
      inline void operator++() noexcept                {--passes_left, current_index += size;}
      inline bool operator!=(const Parallel&) noexcept {return passes_left;}
      inline Parallel& begin() noexcept                {return *this;}
      inline Parallel end() noexcept                   {return Parallel{0};}
    private:
      size_t current_index;
      size_t passes_left;
    public:
      const size_t passes = passes_left;
    };

    template<size_t size>
    class Sequential
    {
    public:
      inline explicit Sequential(const size_t n_elements) noexcept :
        current_index{size ? ((n_elements / size) * size) : 0},
        passes_left{n_elements - current_index}
      {}
      inline size_t operator*() noexcept                 {return current_index;}
      inline void operator++() noexcept                  {--passes_left, ++current_index;}
      inline bool operator!=(const Sequential&) noexcept {return passes_left;}
      inline Sequential& begin() noexcept                {return *this;}
      inline Sequential end() noexcept                   {return Sequential{0};}
    private:
      size_t current_index;
      size_t passes_left;
    public:
      const size_t passes = passes_left;
    };
  }
  
  // not too sure about the portability of this. might have to rethink it.
  template<typename V>
  struct MaskOf
  {
    using Type = bool;
  };
  // template<typename V>
  // using MaskOf = typename SIMD<typename std::remove_reference<decltype(std::declval<V>()[0])>::type>::Mask;


  template<typename T>
  class Array final
  {
  static_assert(std::is_arithmetic<T>::value, "T in Array<T> must be an arithmetic type");
  public:
    inline const T* data(const size_t k = 0) const noexcept
    {
      return array + k;
    }

    inline T* data(const size_t k = 0) noexcept
    {
      return array + k;
    }

    inline size_t size() const noexcept
    {
      return numel;
    }

    inline T operator [](const size_t k) const noexcept
    {
      return array[k];
    }

    inline T& operator [](const size_t k) noexcept
    {
      return array[k];
    }

    inline typename SIMD<T>::Type& as_vector(const size_t k) noexcept
    {
      return (typename SIMD<T>::Type&)(array[k]);
    }   

    inline operator T*() noexcept
    {
      return array;
    }

    inline Array(const size_t number_of_elements) noexcept
    {
      constexpr size_t alignment = SIMD<T>::alignment;
      // early return
      if ((number_of_elements == 0) || (alignment == 0))
      {
        return;
      }
#   if defined(PARALLILOS_HAS_ALIGNED_ALLOC)
      else
      {
        array = reinterpret_cast<T*>(std::aligned_alloc(alignment, number_of_elements * sizeof(T)));
        return;
      }
#   else

      // allocate
      void* memory_block = std::malloc(number_of_elements * sizeof(T) + alignment);
      
      // allocation failure
      if (memory_block == nullptr)
      {
        return;
      }

      // align on alignement boundary
      void* aligned_memory_block = reinterpret_cast<void*>((uintptr_t(memory_block) + alignment) & ~(alignment - 1));

      // bookkeeping of original memory block
      reinterpret_cast<void**>(aligned_memory_block)[-1] = memory_block;
      
      numel = number_of_elements;
      array = reinterpret_cast<T*>(aligned_memory_block);
#   endif
    }

    inline Array(const std::initializer_list<T> initializer_list) :
      Array(initializer_list.size())
    {
      if (array != nullptr)
      {
        size_t k = 0;
        for (T value : initializer_list)
        {
          array[k++] = value;
        }
      }
    }

    inline ~Array() noexcept
    {
#   if defined(PARALLILOS_HAS_ALIGNED_ALLOC)
      if (alignment && array)
      {
        std::free(reinterpret_cast<void**>(array)[-1]);
        return;
      }
#   endif
      std::free(array);
    }
  private:
    T* array     = nullptr;
    size_t numel = 0;
  };

  template<typename T>
  struct SIMD
  {
  static_assert(std::is_arithmetic<T>::value, "T in SIMD<T> must be an arithmetic type");
    static constexpr size_t size      = 0;
    static constexpr size_t alignment = 0;
    using Type = T;
    using Mask = bool;
    static constexpr const char* set = "no SIMD instruction set used for this type";

    static inline Backend::Parallel<size> parallel(const size_t n_elements) noexcept
    {
      return Backend::Parallel<size>{n_elements};
    }
    
    static inline Backend::Sequential<size> sequential(const size_t n_elements) noexcept
    {
      return Backend::Sequential<size>{n_elements};
    }
  };

  // T = type, V = vector type, M = mask type, A = alignment, S = sets used
  #define PARALLILOS_MAKE_SIMD(T, V, M, A, S)                                               \
    template <>                                                                             \
    struct SIMD<T>                                                                          \
    {                                                                                       \
    static_assert(std::is_arithmetic<T>::value, "T in SIMD<T> must be an arithmetic type"); \
      static constexpr size_t size      = sizeof(V)/sizeof(T);                              \
      static constexpr size_t alignment = A;                                                \
      using Type = V;                                                                       \
      using Mask = M;                                                                       \
      static constexpr const char* set = S;                                                 \
      static inline Backend::Parallel<size> parallel(const size_t n_elements) noexcept      \
      {                                                                                     \
        return Backend::Parallel<size>{n_elements};                                         \
      }                                                                                     \
      static inline Backend::Sequential<size> sequential(const size_t n_elements) noexcept  \
      {                                                                                     \
        return Backend::Sequential<size>{n_elements};                                       \
      }                                                                                     \
    };                                                                                      \
    template<>                                                                              \
    struct MaskOf<V>                                                                        \
    {                                                                                       \
      using Type = M;                                                                       \
    }

  // load a vector from unaligned data
  template<typename T>
  PARALLILOS_INLINE auto simd_loadu(const T data[]) -> typename SIMD<T>::Type
  {
    PARALLILOS_TYPE_WARNING(T);
    return *data;
  }

  // load a vector from aligned data
  template<typename T>
  PARALLILOS_INLINE auto simd_loada(const T data[]) -> typename SIMD<T>::Type
  {
    PARALLILOS_TYPE_WARNING(T);
    return *data;
  }

  // store a vector into unaligned memory
  template<typename T>
  PARALLILOS_INLINE void simd_storeu(T addr[], typename SIMD<T>::Type data)
  {
    PARALLILOS_TYPE_WARNING(T, typename SIMD<T>::Type);
    *addr = data;
  }

  // store a vector into aligned memory
  template<typename T>
  PARALLILOS_INLINE void simd_storea(T addr[], typename SIMD<T>::Type data)
  {
    PARALLILOS_TYPE_WARNING(T, typename SIMD<T>::Type);
    *addr = data;
  }

  // load a vector with zeros
  template<typename T>
  PARALLILOS_INLINE auto simd_setzero() -> typename SIMD<T>::Type
  {
    PARALLILOS_TYPE_WARNING(T);
    return 0;
  }

  // load a vector with a specific value
  template<typename T>
  PARALLILOS_INLINE auto simd_setval(const T value) -> typename SIMD<T>::Type
  {
    PARALLILOS_TYPE_WARNING(T);
    return value;
  }
  
  // [a] + [b]
  template<typename V>
  PARALLILOS_INLINE V simd_add(V a, V b)
  {
    PARALLILOS_TYPE_WARNING(V, V);
    return a + b;
  }

  // [a] * [b]
  template<typename V>
  PARALLILOS_INLINE V simd_mul(V a, V b)
  {
    PARALLILOS_TYPE_WARNING(V, V);
    return a * b;
  }

  // [a] - [b]
  template<typename V>
  PARALLILOS_INLINE V simd_sub(V a, V b)
  {
    PARALLILOS_TYPE_WARNING(V, V);
    return a - b;
  }
  
  // [a] / [b]
  template<typename V>
  PARALLILOS_INLINE V simd_div(V a, V b)
  {
    PARALLILOS_TYPE_WARNING(V, V);
    return a / b;
  }
  
  // sqrt([a])
  template<typename V>
  PARALLILOS_INLINE V simd_sqrt(V a)
  { 
    PARALLILOS_TYPE_WARNING(V);
    return std::sqrt(a);
  }

  // [a] + ([b] * [c]) 
  template<typename V>
  PARALLILOS_INLINE V simd_addmul(V a, V b, V c)
  {
    PARALLILOS_TYPE_WARNING(V, V);
    return a + b * c;
  }

  // [a] - ([b] * [c])
  template<typename V>
  PARALLILOS_INLINE V simd_submul(V a, V b, V c)
  {
    PARALLILOS_TYPE_WARNING(V, V, V);
    return a - b * c;
  }

  template <typename V>
  PARALLILOS_INLINE auto simd_eq(V a, V b) -> typename MaskOf<V>::Type
  {
    return a == b;
  }

  template <typename V>
  PARALLILOS_INLINE auto simd_neq(V a, V b) -> typename MaskOf<V>::Type
  {
    return a != b;
  }

  template <typename V>
  PARALLILOS_INLINE auto simd_gt(V a, V b) -> typename MaskOf<V>::Type
  {
    return a > b;
  }

  template <typename T, typename V>
  PARALLILOS_INLINE auto simd_gte(V a, V b) -> typename MaskOf<V>::Type
  {
    return a >= b;
  }

  template <typename V>
  PARALLILOS_INLINE auto simd_lt(V a, V b) -> typename MaskOf<V>::Type
  {
    return a < b;
  }

  template <typename V>
  PARALLILOS_INLINE auto simd_lte(V a, V b) -> typename MaskOf<V>::Type
  {
    return a <= b;
  }

#if defined(PARALLILOS_AVX512F)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD(float, __m512, __mmask16, 64, "AVX512F");
# define PARALLILOS_F32_LOADU(data)           _mm512_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm512_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm512_storeu_ps((void*)addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm512_store_ps((void*)addr, data)
# define PARALLILOS_F32_SETVAL(value)         _mm512_set1_ps(value)
# define PARALLILOS_F32_SETZERO()             _mm512_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm512_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm512_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm512_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm512_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm512_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm512_fmadd_ps(b, c, a)
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm512_fnmadd_ps(a, b, c)

# define PARALLILOS_F32_EQ(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ)
#elif defined(PARALLILOS_FMA)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD(float, __m256, __m256, 32, "AVX, FMA");
# define PARALLILOS_F32_LOADU(data)           _mm256_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm256_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(value)         _mm256_set1_ps(value)
# define PARALLILOS_F32_SETZERO()             _mm256_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm256_fmadd_ps(b, c, a)
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm256_fnmadd_ps(a, b, c)

# define PARALLILOS_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
#elif defined(PARALLILOS_AVX)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD(float, __m256, __m256, 32, "AVX");
# define PARALLILOS_F32_LOADU(data)           _mm256_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm256_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm256_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm256_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(value)         _mm256_set1_ps(value)
# define PARALLILOS_F32_SETZERO()             _mm256_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm256_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm256_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm256_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm256_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm256_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm256_add_ps(a, _mm256_mul_ps(b, c))
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm256_sub_ps(a, _mm256_mul_ps(b, c))

# define PARALLILOS_F32_EQ(a, b)              _mm256_cmp_ps(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm256_cmp_ps(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm256_cmp_ps(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm256_cmp_ps(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm256_cmp_ps(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm256_cmp_ps(a, b, _CMP_LE_OQ)
#elif defined(PARALLILOS_SSE)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD(float, __m128, __m128, 16, "SSE");
# define PARALLILOS_F32_LOADU(data)           _mm_loadu_ps(data)
# define PARALLILOS_F32_LOADA(data)           _mm_load_ps(data)
# define PARALLILOS_F32_STOREU(addr, data)    _mm_storeu_ps(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    _mm_store_ps(addr, data)
# define PARALLILOS_F32_SETVAL(value)         _mm_set1_ps(value)
# define PARALLILOS_F32_SETZERO()             _mm_setzero_ps()
# define PARALLILOS_F32_MUL(a, b)             _mm_mul_ps(a, b)
# define PARALLILOS_F32_ADD(a, b)             _mm_add_ps(a, b)
# define PARALLILOS_F32_SUB(a, b)             _mm_sub_ps(a, b)
# define PARALLILOS_F32_DIV(a, b)             _mm_div_ps(a, b)
# define PARALLILOS_F32_SQRT(a)               _mm_sqrt_ps(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       _mm_add_ps(a, _mm_mul_ps(b, c))
# define PARALLILOS_F32_SUBMUL(a, b, c)       _mm_sub_ps(a, _mm_mul_ps(b, c))

# define PARALLILOS_F32_EQ(a, b)              _mm_cmp_ps(a, b, _CMP_EQ_UQ)
# define PARALLILOS_F32_NEQ(a, b)             _mm_cmp_ps(a, b, _CMP_NEQ_UQ)
# define PARALLILOS_F32_GT(a, b)              _mm_cmp_ps(a, b, _CMP_GT_OQ)
# define PARALLILOS_F32_GTE(a, b)             _mm_cmp_ps(a, b, _CMP_GE_OQ)
# define PARALLILOS_F32_LT(a, b)              _mm_cmp_ps(a, b, _CMP_LT_OQ)
# define PARALLILOS_F32_LTE(a, b)             _mm_cmp_ps(a, b, _CMP_LE_OQ)
#elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
  static_assert(sizeof(float) == 4, "float must be 32 bit");
# define PARALLILOS_F32
  PARALLILOS_MAKE_SIMD(float, float32x4_t, 0, "NEON");
# define PARALLILOS_F32_LOADU(data)           vld1q_f32(data)
# define PARALLILOS_F32_LOADA(data)           vld1q_f32(data)
# define PARALLILOS_F32_STOREU(addr, data)    vst1q_f32(addr, data)
# define PARALLILOS_F32_STOREA(addr, data)    vst1q_f32(addr, data)
# define PARALLILOS_F32_SETVAL(value)         vdupq_n_f32(value)
# define PARALLILOS_F32_SETZERO()             vdupq_n_f32(0.0f)
# define PARALLILOS_F32_MUL(a, b)             vmulq_f32(a, b)
# define PARALLILOS_F32_ADD(a, b)             vaddq_f32(a, b)
# define PARALLILOS_F32_SUB(a, b)             vsubq_f32(a, b)
# define PARALLILOS_F32_DIV(a, b)             vdivq_f32(a, b)
# define PARALLILOS_F32_SQRT(a)               vsqrtq_f32(a)
# define PARALLILOS_F32_ADDMUL(a, b, c)       vmlaq_f32(a, b, c)
# define PARALLILOS_F32_SUBMUL(a, b, c)       vmlsq_f32(a, b, c)
#endif

#ifdef PARALLILOS_F32
  template<>
  PARALLILOS_INLINE auto simd_setzero<float>() -> SIMD<float>::Type
  {
    return PARALLILOS_F32_SETZERO();
# undef PARALLILOS_F32_SETZERO
  }

  template <>
  PARALLILOS_INLINE auto simd_loadu(const float data[]) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_LOADU(data);
# undef PARALLILOS_F32_LOADU
  }

  template <>
  PARALLILOS_INLINE auto simd_loada(const float data[]) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_LOADA(data);
# undef PARALLILOS_F32_LOADA
  }

  template <>
  PARALLILOS_INLINE void simd_storeu(float addr[], SIMD<float>::Type data)
  {
    PARALLILOS_F32_STOREU(addr, data);
# undef PARALLILOS_F32_STOREU
  }

  template <>
  PARALLILOS_INLINE void simd_storea(float addr[], SIMD<float>::Type data)
  {
    PARALLILOS_F32_STOREA(addr, data);
# undef PARALLILOS_F32_STOREA
  }

  template <>
  PARALLILOS_INLINE auto simd_setval(const float value) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_SETVAL(value);
# undef PARALLILOS_F32_SETVAL
  }

  template <>
  PARALLILOS_INLINE auto simd_add(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_ADD(a, b);
# undef PARALLILOS_F32_ADD
  }

  template <>
  PARALLILOS_INLINE auto simd_mul(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_MUL(a, b);
# undef PARALLILOS_F32_MUL
  }

  template <>
  PARALLILOS_INLINE auto simd_sub(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_SUB(a, b);
# undef PARALLILOS_F32_SUB
  }

  template <>
  PARALLILOS_INLINE auto simd_div(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_DIV(a, b);
# undef PARALLILOS_F32_DIV
  }

  template <>
  PARALLILOS_INLINE auto simd_sqrt(SIMD<float>::Type a) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_SQRT(a);
# undef PARALLILOS_F32_SQRT
  }

  template <>
  PARALLILOS_INLINE auto simd_addmul(SIMD<float>::Type a, SIMD<float>::Type b, SIMD<float>::Type c) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_ADDMUL(a, b, c);
#   undef PARALLILOS_F32_ADDMUL
  }

  template <>
  PARALLILOS_INLINE auto simd_submul(SIMD<float>::Type a, SIMD<float>::Type b, SIMD<float>::Type c) -> SIMD<float>::Type
  {
    return PARALLILOS_F32_SUBMUL(a, b, c);
# undef PARALLILOS_F32_SUBMUL
  }

  template <>
  PARALLILOS_INLINE auto simd_eq<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_EQ(a, b);
# undef PARALLILOS_F32_EQ
  }

  template <>
  PARALLILOS_INLINE auto simd_neq<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_NEQ(a, b);
# undef PARALLILOS_F32_NEQ
  }

  template <>
  PARALLILOS_INLINE auto simd_gt<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_GT(a, b);
#   undef PARALLILOS_F32_GT
  }

  template <>
  PARALLILOS_INLINE auto simd_gte<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_GTE(a, b);
# undef PARALLILOS_F32_GTE
  }

  template <>
  PARALLILOS_INLINE auto simd_lt<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_LT(a, b);
#   undef PARALLILOS_F32_LT
  }

  template <>
  PARALLILOS_INLINE auto simd_lte<SIMD<float>::Type>(SIMD<float>::Type a, SIMD<float>::Type b) -> SIMD<float>::Mask
  {
    return PARALLILOS_F32_LTE(a, b);
# undef PARALLILOS_F32_LTE
  }
#endif

#if defined(PARALLILOS_AVX512F)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD(double, __m512d, __m512d, 64, "AVX512F");
# define PARALLILOS_F64_LOADU(data)           _mm512_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm512_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm512_storeu_pd((void*)addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm512_store_pd((void*)addr, data)
# define PARALLILOS_F64_SETVAL(value)         _mm512_set1_pd(value)
# define PARALLILOS_F64_SETZERO()             _mm512_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm512_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm512_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm512_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm512_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm512_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm512_fmadd_pd(b, c, a)
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm512_fnmadd_pd(a, b, c)
#elif defined(PARALLILOS_FMA)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD(double, __m256d, __m256d, 32, "AVX, FMA");
# define PARALLILOS_F64_LOADU(data)           _mm256_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm256_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(value)         _mm256_set1_pd(value)
# define PARALLILOS_F64_SETZERO()             _mm256_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm256_fmadd_pd(b, c, a)
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm256_fnmadd_pd(a, b, c)
#elif defined(PARALLILOS_AVX)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD(double, __m256d, __m256d, 32, "AVX");
# define PARALLILOS_F64_LOADU(data)           _mm256_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm256_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm256_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm256_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(value)         _mm256_set1_pd(value)
# define PARALLILOS_F64_SETZERO()             _mm256_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm256_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm256_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm256_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm256_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm256_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm256_add_pd(a, _mm256_mul_pd(b, c))
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm256_sub_pd(a, _mm256_mul_pd(b, c))
#elif defined(PARALLILOS_SSE2)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD(double, __m128d, __m128d, 16, "SSE2");
# define PARALLILOS_F64_LOADU(data)           _mm_loadu_pd(data)
# define PARALLILOS_F64_LOADA(data)           _mm_load_pd(data)
# define PARALLILOS_F64_STOREU(addr, data)    _mm_storeu_pd(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    _mm_store_pd(addr, data)
# define PARALLILOS_F64_SETVAL(value)         _mm_set1_pd(value)
# define PARALLILOS_F64_SETZERO()             _mm_setzero_pd()
# define PARALLILOS_F64_MUL(a, b)             _mm_mul_pd(a, b)
# define PARALLILOS_F64_ADD(a, b)             _mm_add_pd(a, b)
# define PARALLILOS_F64_SUB(a, b)             _mm_sub_pd(a, b)
# define PARALLILOS_F64_DIV(a, b)             _mm_div_pd(a, b)
# define PARALLILOS_F64_SQRT(a)               _mm_sqrt_pd(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       _mm_add_pd(a, _mm_mul_pd(b, c))
# define PARALLILOS_F64_SUBMUL(a, b, c)       _mm_sub_pd(a, _mm_mul_pd(b, c))
#elif defined(PARALLILOS_NEON) || defined(PARALLILOS_NEON64)
# define PARALLILOS_F64
  static_assert(sizeof(double) == 8, "float must be 64 bit");
  PARALLILOS_MAKE_SIMD(double, float64x4_t, 0, "NEON");
# define PARALLILOS_F64_LOADU(data)           vld1q_f64(data)
# define PARALLILOS_F64_LOADA(data)           vld1q_f64(data)
# define PARALLILOS_F64_STOREU(addr, data)    vst1q_f64(addr, data)
# define PARALLILOS_F64_STOREA(addr, data)    vst1q_f64(addr, data)
# define PARALLILOS_F64_SETVAL(value)         vdupq_n_f64(value)
# define PARALLILOS_F64_SETZERO()             vdupq_n_f64(0.0)
# define PARALLILOS_F64_MUL(a, b)             vmulq_f64(a, b)
# define PARALLILOS_F64_ADD(a, b)             vaddq_f64(a, b)
# define PARALLILOS_F64_SUB(a, b)             vsubq_f64(a, b)
# define PARALLILOS_F64_DIV(a, b)             vdivq_f64(a, b)
# define PARALLILOS_F64_SQRT(a)               vsqrtq_f64(a)
# define PARALLILOS_F64_ADDMUL(a, b, c)       vmlaq_f64(a, b, c)
# define PARALLILOS_F64_SUBMUL(a, b, c)       vmlsq_f64(a, b, c)
#endif

#ifdef PARALLILOS_F64
  template<>
  PARALLILOS_INLINE auto simd_setzero<double>() -> SIMD<double>::Type
  {
    return PARALLILOS_F64_SETZERO();
# undef PARALLILOS_F64_SETZERO
  }

  template <>
  PARALLILOS_INLINE auto simd_loadu(const double data[]) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_LOADU(data);
# undef PARALLILOS_F64_LOADU
  }

  template <>
  PARALLILOS_INLINE auto simd_loada(const double data[]) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_LOADA(data);
# undef PARALLILOS_F64_LOADA
  }

  template <> 
  PARALLILOS_INLINE void simd_storeu(double addr[], SIMD<double>::Type data)
  {
    PARALLILOS_F64_STOREU(addr, data);
# undef PARALLILOS_F64_STOREU
  }

  template <>
  PARALLILOS_INLINE void simd_storea(double addr[], SIMD<double>::Type data)
  {
    PARALLILOS_F64_STOREA(addr, data);
# undef PARALLILOS_F64_STOREA
  }

  template <>
  PARALLILOS_INLINE auto simd_setval(const double value) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_SETVAL(value);
# undef PARALLILOS_F64_SETVAL
  }

  template <>
  PARALLILOS_INLINE auto simd_add(const SIMD<double>::Type a, const SIMD<double>::Type b) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_ADD(a, b);
# undef PARALLILOS_F64_ADD
  }

  template <>
  PARALLILOS_INLINE auto simd_mul(SIMD<double>::Type a, SIMD<double>::Type b) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_MUL(a, b);
# undef PARALLILOS_F64_MUL
  }

  template <>
  PARALLILOS_INLINE auto simd_sub(SIMD<double>::Type a, SIMD<double>::Type b) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_SUB(a, b);
# undef PARALLILOS_F64_SUB
  }

  template <>
  PARALLILOS_INLINE auto simd_div(SIMD<double>::Type a, SIMD<double>::Type b) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_DIV(a, b);
# undef PARALLILOS_F64_DIV
  }

  template <>
  PARALLILOS_INLINE auto simd_sqrt(SIMD<double>::Type a) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_SQRT(a);
# undef PARALLILOS_F64_SQRT
  }

  template <>
  PARALLILOS_INLINE auto simd_addmul(SIMD<double>::Type a, SIMD<double>::Type b, SIMD<double>::Type c) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_ADDMUL(a, b, c);
# undef PARALLILOS_F64_ADDMUL
  }

  template <>
  PARALLILOS_INLINE auto simd_submul(SIMD<double>::Type a, SIMD<double>::Type b, SIMD<double>::Type c) -> SIMD<double>::Type
  {
    return PARALLILOS_F64_SUBMUL(a, b, c);
# undef PARALLILOS_F64_SUBMUL
  }
#endif

#if defined(PARALLILOS_AVX512F)
# define PARALLILOS_I32
  static_assert(sizeof(int32_t) == 4, "int32_t must be 32 bit");
  PARALLILOS_MAKE_SIMD(int32_t, __m512i, __m512i, 64, "AVX512F");
# define PARALLILOS_I32_LOADU(data)           _mm512_loadu_si512(data)
# define PARALLILOS_I32_LOADA(data)           _mm512_load_si512(data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm512_storeu_si512((void*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm512_store_si512((void*)addr, data)
# define PARALLILOS_I32_SETVAL(value)         _mm512_set1_epi32(value)
# define PARALLILOS_I32_SETZERO()             _mm512_setzero_si512()
# define PARALLILOS_I32_MUL(a, b)             _mm512_mullo_epi32 (a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm512_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm512_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm512_cvtps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(a), _mm512_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm512_cvtps_epi32(_mm512_sqrt_ps(_mm512_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm512_add_epi32(a, _mm512_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm512_sub_epi32(a, _mm512_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm512_div_epi32(a, b)
# endif
#elif defined(PARALLILOS_AVX2)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD(int32_t, __m256i, __m256i, 32, "AVX2, AVX");
# define PARALLILOS_I32_LOADU(data)           _mm256_loadu_si256((const __m256i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm256_load_si256((const __m256i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm256_storeu_si256 ((__m256i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm256_store_si256((__m256i*)addr, data)
# define PARALLILOS_I32_SETVAL(value)         _mm256_set1_epi32(value)
# define PARALLILOS_I32_SETZERO()             _mm256_setzero_si256()
# define PARALLILOS_I32_MUL(a, b)             _mm256_mullo_epi32(a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm256_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm256_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm256_cvtps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(a), _mm256_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm256_add_epi32(a, _mm256_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm256_sub_epi32(a, _mm256_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm256_div_epi32(a, b)
# endif
#elif defined(PARALLILOS_SSE4_1)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD(int32_t, __m128i, __m128i, 16, "SSE4.1, SSE2, SSE");
# define PARALLILOS_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define PARALLILOS_I32_SETVAL(value)         _mm_set1_epi32(value)
# define PARALLILOS_I32_SETZERO()             _mm_setzero_si128()
# define PARALLILOS_I32_MUL(a, b)             _mm_mullo_epi32(a, b)
# define PARALLILOS_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_mul_epi32(b, c))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_mul_epi32(b, c))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm_div_epi32(a, b)
# endif
#elif defined(PARALLILOS_SSE2)
# define PARALLILOS_I32
  PARALLILOS_MAKE_SIMD(int32_t, __m128i, __m128i, 16, "SSE2, SSE");
# define PARALLILOS_I32_LOADU(data)           _mm_loadu_si128((const __m128i*)data)
# define PARALLILOS_I32_LOADA(data)           _mm_load_si128((const __m128i*)data)
# define PARALLILOS_I32_STOREU(addr, data)    _mm_storeu_si128((__m128i*)addr, data)
# define PARALLILOS_I32_STOREA(addr, data)    _mm_store_si128((__m128i*)addr, data)
# define PARALLILOS_I32_SETVAL(value)         _mm_set1_epi32(value)
# define PARALLILOS_I32_SETZERO()             _mm_setzero_si128()
# define PARALLILOS_I32_MUL(a, b)             _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_ADD(a, b)             _mm_add_epi32(a, b)
# define PARALLILOS_I32_SUB(a, b)             _mm_sub_epi32(a, b)
# define PARALLILOS_I32_DIV(a, b)             _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# define PARALLILOS_I32_SQRT(a)               _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)))
# define PARALLILOS_I32_ADDMUL(a, b, c)       _mm_add_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
# define PARALLILOS_I32_SUBMUL(a, b, c)       _mm_sub_epi32(a, _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(b), _mm_cvtepi32_ps(c))))
# if defined(PARALLILOS_SVML)
#   undef  PARALLILOS_I32_DIV
#   define PARALLILOS_I32_DIV(a, b)           _mm_cvtps_epi32(_mm_div_ps(_mm_cvtepi32_ps(a), _mm_cvtepi32_ps(b)))
# endif
#endif
  
#ifdef PARALLILOS_I32
  template<>
  PARALLILOS_INLINE auto simd_setzero<int32_t>(void) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_SETZERO();
# undef PARALLILOS_I32_SETZERO
  }

  template <>
  PARALLILOS_INLINE auto simd_loadu(const int32_t data[]) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_LOADU(data);
# undef PARALLILOS_I32_LOADU
  }

  template <>
  PARALLILOS_INLINE auto simd_loada(const int32_t data[]) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_LOADA(data);
# undef PARALLILOS_I32_LOADA
  }

  template <>
  PARALLILOS_INLINE void simd_storeu(int32_t addr[], SIMD<int32_t>::Type data)
  {
    PARALLILOS_I32_STOREU(addr, data);
# undef PARALLILOS_I32_STOREU
  }

  template <>
  PARALLILOS_INLINE void simd_storea(int32_t addr[], SIMD<int32_t>::Type data)
  {
    PARALLILOS_I32_STOREA(addr, data);
# undef PARALLILOS_I32_STOREA
  }

  template <>
  PARALLILOS_INLINE auto simd_setval(const int32_t value) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_SETVAL(value);
# undef PARALLILOS_I32_SETVAL
  }

  template <>
  PARALLILOS_INLINE auto simd_add(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_ADD(a, b);
# undef PARALLILOS_I32_ADD
  }

  template <>
  PARALLILOS_INLINE auto simd_mul(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_MUL(a, b);
# undef PARALLILOS_I32_MUL
  }

  template <>
  PARALLILOS_INLINE auto simd_sub(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_SUB(a, b);
# undef PARALLILOS_I32_SUB
  }

  template <>
  PARALLILOS_INLINE auto simd_div(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_DIV(a, b);
# undef PARALLILOS_I32_DIV
  }

  template <>
  PARALLILOS_INLINE auto simd_sqrt(SIMD<int32_t>::Type a) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_SQRT(a);
# undef PARALLILOS_I32_SQRT
  }

  template <>
  PARALLILOS_INLINE auto simd_addmul(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b, SIMD<int32_t>::Type c) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_ADDMUL(a, b, c);
# undef PARALLILOS_I32_ADDMUL
  }

  template <>
  PARALLILOS_INLINE auto simd_submul(SIMD<int32_t>::Type a, SIMD<int32_t>::Type b, SIMD<int32_t>::Type c) -> SIMD<int32_t>::Type
  {
    return PARALLILOS_I32_SUBMUL(a, b, c);
# undef PARALLILOS_I32_SUBMUL
  }
#endif
}
#undef PARALLILOS_TYPE_WARNING
#endif