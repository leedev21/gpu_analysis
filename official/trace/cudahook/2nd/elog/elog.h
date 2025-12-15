#pragma once


/**
  * new environment log control grammar
  *       ELOG_LEVEL_{GLOBAL | MODULENAME} = {off | fatal | err | warn | info | debug | hints | trace}
  *       ELOG_SINK_{GLOBAL | MODULENAME} = {stdout | stderr | file | syslog}
  *       ELOG_FILEPATH_{GLOBAL} = {filepath}
  *       ELOG_FILENUM_{GLOBAL} = {filenum}
  *       ELOG_FILESIZE_{GLOBAL} = {filesize}
  *       ELOG_PERSISTENT_{GLOBAL} = {filepath}
  * e.g: 
  *     export ELOG_LEVEL_sampleA=info
  *     export ELOG_LEVEL_ELOG=debug
  * 
  * legacy environment log control grammar compatibility
  */

#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <ostream>
#include <vector>
#include <sstream>
#include <memory.h>
#include <functional>

namespace elog {

#if !defined(_MSC_VER)
  #define ELOG_EXPORT __attribute((visibility("default")))
  #define ELOG_NORETURN __attribute__((noreturn))
  #define ELOG_INIT_PRIORITY __attribute__((init_priority(111)))
#else
  #define ELOG_EXPORT
  #define ELOG_NORETURN __declspec(noreturn)
  #define ELOG_INIT_PRIORITY
#endif

ELOG_EXPORT bool init();

/**
  * @enum level
  * @brief log level
  */
enum ELOG_EXPORT level : int32_t {
  off   = 0x00, /**< log is disabled */
  fatal = 0x01, /**< severe error, application abort and output backtrace */
  err   = 0x02, /**< severe error, output backtrace */
  warn  = 0x04, /**< potentially harmful situations */
  info  = 0x08, /**< messages that highlight the progress */
  debug = 0x10, /**< information that arg mot useful to debug */
  hints = 0x20, /**< potentially improve performance */
  trace = 0x40, /**< API trace with parameters */
};

/**
  * @enum sink_mask
  * @brief mask of destination
  */
enum ELOG_EXPORT sink_mask : int32_t {
  std_out = 0x01, /**< stdout */
  file    = 0x02, /**< file */
  std_err = 0x04, /**< stderr */
  sys_log = 0x08, /**< syslog */
};

struct env_entry {
  #define ENV_LOG_LEVEL_OFF   "off"
  #define ENV_LOG_LEVEL_FATAL "fatal"
  #define ENV_LOG_LEVEL_ERR   "err"
  #define ENV_LOG_LEVEL_WARN  "warn"
  #define ENV_LOG_LEVEL_INFO  "info"
  #define ENV_LOG_LEVEL_DEBUG "debug"
  #define ENV_LOG_LEVEL_HINTS "hints"
  #define ENV_LOG_LEVEL_TRACE "trace"

  #define ENV_LOG_DEST_STDOUT "stdout"
  #define ENV_LOG_DEST_STDERR "stderr"

  char enable_flag_[32];      /**< flag string of enable log, e.g "1" */
  char disable_flag_[32];     /**< flag string of diable log, e.g "0" */
  struct env_level {
    char name_[128];          /**< name of env, read by getenv() */
    char type_[32];           /**< type string of env */
  } env_level_[32];           /**< maximun of events array */
  int num_of_env_level_;      /**< number of event set by user */
  char env_module_name_[128]; /**< env name of module list */
  char env_dest_name_[128];   /**< env name of destination */
  bool bitmask_mode_;         /**< true - bitmask mode, false - cover mode */
  bool prefix_enable_;        /**< true - enable prefix, 
                                  false - disable prefix */
};

/**
 * @brief register new group, it's an option for module
 * @param[in]  groupname    name of group
 * @param[in]  entries      entries
 * @param[in]  num_of_entry number of entries
 * @return >=0 if success, -1 if failed
 */
ELOG_EXPORT int register_group(const char* groupname, env_entry *entries);

/**
  * @brief register new module
  * @param[in]  modulename  name of module
  * @param[in]  lvl         out level
  * @return >=0 if success, -1 if failed
  */
ELOG_EXPORT int register_module(const char* modulename,
        level lvl = level::info, int group_handle = -1);

/**
  * @brief get module handle ID by module name in log
  * @param[in]  modulename  name of module
  * @return >=0 if success, -1 if failed
  */
ELOG_EXPORT int get_module_handle(const char* modulename);

/**
  * @brief set new level in runtime
  * @param[in]  handle      handle return from @see register_module
  * @param[in]  lvl         level
  * @return void
  */
ELOG_EXPORT void set_level(int handle, level lvl);

/**
  * @brief enable new log level
  * @param[in]  handle      handle return from @see register_module
  * @param[in]  lvl         level
  * @return void
  */
ELOG_EXPORT void set_level_mask(int handle, level lvl);

/**
  * @brief get log level in runtime
  * @param[in]  handle      handle return from @see register_module
  * @return @see level
  */
ELOG_EXPORT level get_level(int handle);

/**
  * @brief check module log level status
  * @param[in]  handle      handle return from @see register_module
  * @param[in]  lvl         level to check
  * @return     ture - on, false - off
  */
ELOG_EXPORT bool module_is_on(int handle, level lvl);

/**
  * @brief set log destination
  * param[in] handle    the module handle return from @see register_module
  * param[in] sink      destination @see sink_mask
  * parma[in] filename  the file parth and name
  * @return true - success, false - failure
  */
ELOG_EXPORT bool set_sink(sink_mask sink, const char* filename = nullptr);
ELOG_EXPORT bool set_sink(int handle, sink_mask sink, const char* filename = nullptr);

/**
  * @brief set system pattern
  * @param[in]  pattern     string of pattern,
  *                         e.g "%Y-%m-%d %H:%M:%S.%f:%L [T %t]%v: "
  * @return void
  * @table
  * | flag | meaning | example |
  * | :--: | :--: | :--: |
  * |%v| The actual text to log | "some user text"
  * |%t| Thread id | "1232"
  * |%P| Process id | "3456"
  * |%n| Logger's name | "some logger name"
  * |%l| The log level of the message | "debug", "info", etc
  * |%L| Short log level of the message | "D", "I", etc
  * |%a| Abbreviated weekday name | "Thu"
  * |%A| Full weekday name | "Thursday"
  * |%b| Abbreviated month name | "Aug"
  * |%B| Full month name | "August"
  * |%c| Date and time representation | "Thu Aug 23 15:35:46 2014"
  * |%C| Year in 2 digits | "14"
  * |%Y| Year in 4 digits | "2014"
  * |%D| or %x Short MM/DD/YY date | "08/23/14"
  * |%m| Month 01-12 | "11"
  * |%d| Day of month 01-31 | "29"
  * |%H| Hours in 24 format 00-23 | "23"
  * |%I| Hours in 12 format 01-12 | "11"
  * |%M| Minutes 00-59 | "59"
  * |%S| Seconds 00-59 | "58"
  * |%e| Millisecond part of the current second 000-999 | "678"
  * |%f| Microsecond part of the current second 000000-999999 | "056789"
  * |%F| Nanosecond part of the current second 000000000-999999999 | "256789123"
  * |%p| AM/PM | "AM"
  * |%r| 12 hour clock | "02:55:02 PM"
  * |%R| 24-hour HH:MM time, equivalent to %H:%M | "23:55"
  * |%T| or %X ISO 8601 time format (HH:MM:SS),
  *      equivalent to %H:%M:%S | "23:55:59"
  * |%z| ISO 8601 offset from UTC in timezone ([+/-]HH:MM) | "+02:00"
  * |%E| Seconds since the epoch | "1528834770"
  * |%%| The % sign | "%"
  * |%+| spdlog's default format | "[2014-10-31 23:46:59.678] [mylogger] [info]
  *      Some message"
  * |%^| start color range (can be used only once) | "[mylogger] [info(green)]
  *      Some message"
  * |%$| end color range (for example %^[+++]%$ %v)
  *      (can be used only once) | [+++] Some message
  * |%@| Source file and line Same as %g:%# | /some/dir/my_file.cpp:123
  * |%s| Basename of the source file | my_file.cpp
  * |%g| Full or relative path of the source file as appears in
  *      the __FILE__ macro | /some/dir/my_file.cpp
  * |%#| Source line | 123
  * |%!| Source function etc. see tweakme for pretty-print) | my_func
  * |%o| Elapsed time in milliseconds since previous message | 456
  * |%i| Elapsed time in microseconds since previous message | 456
  * |%u| Elapsed time in nanoseconds since previous message | 11456
  * |%O| Elapsed time in seconds since previous message | 4
  * @endtable
  */
ELOG_EXPORT void set_pattern(const char* pattern);

/**
  * @brief get system pattern
  * @return !=nullptr the current pattern, ==nullptr failure
  */
ELOG_EXPORT const char* get_pattern();

/**
  * @brief printf style interface
  * @param[in] handle    module handle return from register @see register_module
  * @param[in] *filename code filename
  * @param[in] line      code line
  * @param[in] lvl       log level
  * @param[in] *format   paramters format
  * @param[in] ...       parameters
  * @return void
  */
ELOG_EXPORT void eprintf(int handle, const char* filename, int line,
      level lvl, const char* format, ...);

/**
  * @brief dump interface, design for large IR log, zero copy sendto
  *         destination without pattern
  * @param[in] handle     module handle return from register
  *                       @see register_module
  * @param[in] *filename  code filename
  * @param[in] line       code line
  * @param[in] lvl        log level
  * @param[in] *data      data
  * @param[in] size       parameters
  * @param[in] fmt        printf-linnk format
  * @param[in] ...        variable parameter
  * @return void
  */
ELOG_EXPORT void edump(int handle, const char* filename, int line,
      level lvl, const char* data, size_t size, const char* format, ...);

/**
  * @brief backtrace
  * @return void
  */
ELOG_EXPORT void ebacktrace();

/**
  * @brief printf style interface
  * @param[in] *filename code filename
  * @param[in] line      code line
  * @param[in] *format   paramters format
  * @param[in] ...       parameters
  * @return void
  */
ELOG_EXPORT void ebacktrace(int handle, const char* filename, int line,
      level lvl, const char* format, ...);

/**
  * @brief flush all log data to destination
  * @return void
  */
ELOG_EXPORT void eflush();

typedef std::function<void(const char* msg, size_t len)>
  ecustom_log_callback_fn;
/**
  * @brief register callback function for log output
  * @param[in] callback  callback function
  * @return void
  * @note  the callback function shall be call before register_module, and 
  *        thread safe.
  */
ELOG_EXPORT void ecustom_log_callback(const ecustom_log_callback_fn &callback);

/**
  * @class elog_base, base class for user inherit, override functions to control
  *   switch environments, header data, file line data and output destinaiton.
  */
class ELOG_EXPORT elog_base {
public:
  /**
    * @brief elog_base constructor of elog_base
    * @param[in] name   module name
    */
  ELOG_EXPORT explicit elog_base(const char* name);
  elog_base() = delete;
  ELOG_EXPORT virtual ~elog_base();

public:
  /**
    * @brief set_sink set output destination by elog
    * @param[in] sink     output destination, @see sink_mask
    * @param[in] filename optional full file path and name
    * @return             true - success, flase - failure
    */
  ELOG_EXPORT bool  set_sink(sink_mask sink, const char * filename = nullptr);

  /**
    * @brief set runtime level
    * @param[in] lvl  runtime log level @see level
    * @return         true - success, flase - failure
    */
  ELOG_EXPORT bool  set_level(level lvl);

  /**
    * @brief set_level_mask enable one or multiple level
    * @param[in] lvl  runtime log level @see level
    * @return         true - success, flase - failure
    */
  ELOG_EXPORT bool  set_level_mask(level lvl);

  /**
    * @brief get_level_short_name get the 1st charactor of level name
    * @param[in] lvl  runtime log level @see level
    * @return         the 1st charactor of level name
    */
  ELOG_EXPORT char get_level_short_name(level lvl);

 public:
  /**
   * @brief override is_on function to enable/disable output on the level.
   * @param[in] lvl  runtime log level, @see level
   */
  ELOG_EXPORT virtual bool is_on(level lvl);

  /**
    * @brief override fill_header function to set user's header.
    * @param[out] data  buffer for record user's header.
    * @param[in]  size  size of data.
    * @param[in]  lvl   runtime level.
    * @return           size of user output.
    */
  ELOG_EXPORT virtual int  fill_header(char* data, int size, level lvl);

  /**
    * @brief override fill_line function to set user's file:line.
    * @param[out] oss       ostream for user output.
    * @param[in]  filename  runtime __FILE__.
    * @param[in]  line      runtime __LINE__.
    * @return               size of user output.
    */
  ELOG_EXPORT virtual int  fill_line(std::ostream& oss,
                          const char* filename, int line);

  /**
    * @brief override output for user's log destination.
    * @param[in]  data  full log data.
    * @param[in]  size  data size.
    * @return           size of user output.
    */
  ELOG_EXPORT virtual int output(const char* data, int size);

public:
  ELOG_EXPORT int handle_ = -1;

protected:
  ELOG_EXPORT static const char 
  log_level_short_name_[static_cast<int>(elog::level::trace) + 1];

private:
  bool bitmask_level_ = false;
};

/**
  * @class emessage, it receives streaming message input, for the calling and
  *         parameters 
  * of this class @see ECOUT
  */
class ELOG_EXPORT emessage {
public:
  /**
    * @brief emessage construct
    * @param[in] handle   module handle return from register
    *                     @see register_module
    * @param[in] *file    code filename
    * @param[in] line     code line
    * @param[in] lvl      log level
    */
  explicit emessage(int handle, const char* file, int line, level lvl);
  explicit emessage(elog_base *handle, const char* file, int line, level lvl);
  emessage(const emessage&) = delete;
  void operator=(const emessage&) = delete;
  virtual ~emessage();

  /**
   * @brief the ostream object to receive << message.
   * @return reference of ostream
   */
  ELOG_EXPORT std::ostream& stream();

  ELOG_EXPORT void flush_out_data();
  ELOG_EXPORT int preserved_errno() const;

  static const size_t max_message_len;
  struct emessagedata;

protected:
  /**
    * @brief emessage constuct for check/fatal message
    * @param[in] handle   module handle return from register
    *                     @see register_module
    * @param[in] *file    code filename
    * @param[in] line     code line
    * @param[in] lvl      log level
    * @param[in] cond     conditional formula
    */
  emessage(int handle, const char* file, int line, level lvl, const char* cond);
  emessage(elog_base* handle, const char* file, int line, level lvl, const char* cond);

private:
  void init(int handle, const char* file, int line,
            level lvl, const char* cond = nullptr);
  void init(elog_base* handle, const char* file, int line,
            level lvl, const char* cond = nullptr);
  const char* const_basename(const char* filepath);
  void send_to_sink();

  emessagedata* allocated_ = nullptr; /**< the data in a separate struct so
                                           that each instance of
                                          emessage uses less stack space.*/
  emessagedata* data_ = nullptr;

  int32_t sink_mask_ = static_cast<int32_t>(sink_mask::std_out);
};

/**
  * @class fatal message class
  */
class ELOG_EXPORT efatal_message : public emessage {
public:
  efatal_message(int handle, const char* file, int line,
                level lvl, const char* cond);
  efatal_message(int handle, const char* file, int line, level lvl);
  efatal_message(const char* file, int line, const char* cond);
  efatal_message(const char* file, int line, const char* cond, bool isfree);
};

template <class T>
inline const T& GetReferenceableValue(const T& t) { return t; }
inline char           GetReferenceableValue(char               t) { return t; }
inline unsigned char  GetReferenceableValue(unsigned char      t) { return t; }
inline signed char    GetReferenceableValue(signed char        t) { return t; }
inline short          GetReferenceableValue(short              t) { return t; }
inline unsigned short GetReferenceableValue(unsigned short     t) { return t; }
inline int            GetReferenceableValue(int                t) { return t; }
inline unsigned int   GetReferenceableValue(unsigned int       t) { return t; }
inline long           GetReferenceableValue(long               t) { return t; }
inline unsigned long  GetReferenceableValue(unsigned long      t) { return t; }
#if __cplusplus >= 201103L
inline long long      GetReferenceableValue(long long          t) { return t; }
inline unsigned long long GetReferenceableValue(unsigned long long t) {
  return t;
}
#endif

template <typename T>
inline void MakeCheckOpValueString(std::ostream* os, const T& v) {
  (*os) << v;
}

template <>
void inline MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  }
  else {
    (*os) << "char value " << static_cast<short>(v);
  }
}
template <>
void inline MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  }
  else {
    (*os) << "signed char value " << static_cast<short>(v);
  }
}
template <>
void inline MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  }
  else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);
  }
}

// This is required because nullptr is only present in c++ 11 and later.
#if 1 && __cplusplus >= 201103L
// Provide printable value for nullptr_t
template <>
void inline MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (void)(v);  // Avoid unused variable warning (nullptr_t isn't used on its own)
  (*os) << "nullptr";
}
#endif


class CheckOpMessageBuilder {
public:
  explicit CheckOpMessageBuilder(const char* exprtext) {
    stream_ << exprtext << " (";
  }
  std::ostream* ForVar1() {
    return &stream_;
  }
  std::ostream* ForVar2() {
    stream_ << " vs. ";
    return &stream_;
  }
  char* NewString() {
    stream_ << ")";
    char* ret = new char[stream_.str().size() + 1];
    memset(ret, 0, stream_.str().size() + 1);
    memcpy(ret, stream_.str().c_str(), stream_.str().size());
    return ret;
  }

private:
  std::ostringstream stream_;
};

template <typename T1, typename T2>
char* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext) {
  CheckOpMessageBuilder comb(exprtext);
  MakeCheckOpValueString(comb.ForVar1(), v1);
  MakeCheckOpValueString(comb.ForVar2(), v2);
  return comb.NewString();
}

#define E_DEFINE_CHECK_OP_IMPL(name, op) \
  template <typename T1, typename T2> \
  inline char* name##Impl(const T1& v1, const T2& v2,    \
                            const char* exprtext) { \
    if (v1 op v2) \
      return nullptr; \
    else \
      return MakeCheckOpString(v1, v2, exprtext); \
  } \
  inline char* name##Impl(int v1, int v2, const char* exprtext) { \
    return name##Impl<int, int>(v1, v2, exprtext); \
  }

E_DEFINE_CHECK_OP_IMPL(Check_EQ, == )
E_DEFINE_CHECK_OP_IMPL(Check_NE, != )
E_DEFINE_CHECK_OP_IMPL(Check_LE, <= )
E_DEFINE_CHECK_OP_IMPL(Check_LT, < )
E_DEFINE_CHECK_OP_IMPL(Check_GE, >= )
E_DEFINE_CHECK_OP_IMPL(Check_GT, > )
#undef E_DEFINE_CHECK_OP_IMPL

#define ECHECK_OP_LOG(name, op, val1, val2, log)   \
  while (char* _result =                          \
         elog::Check##name##Impl(                 \
             elog::GetReferenceableValue(val1),   \
             elog::GetReferenceableValue(val2),   \
             "check failed: " #val1 " " #op " " #val2))            \
    log(__FILE__, __LINE__, _result).stream()

#define ECHECK_OP(name, op, val1, val2) \
  ECHECK_OP_LOG(name, op, val1, val2, elog::efatal_message)

} // namespace elog


#define EOFF    elog::level::off
#define EFATAL  elog::level::fatal
#define EERR    elog::level::err
#define EWARN   elog::level::warn
#define EINFO   elog::level::info
#define EDEBUG  elog::level::debug
#define EHINTS  elog::level::hints
#define ETRACE  elog::level::trace

/**
  * @brief macro of stream message output
  * @param[in] handle   module handle return from register @see register_module
  * @param[in] lvl      log level
  */
#define ECOUT(handle, lvl) \
  if(elog::module_is_on(handle, lvl)) \
    elog::emessage(handle, __FILE__, __LINE__, lvl).stream()

#define ECOUT_F(handle) ECOUT(handle, EFATAL)
#define ECOUT_E(handle) ECOUT(handle, EERR)
#define ECOUT_W(handle) ECOUT(handle, EWARN)
#define ECOUT_I(handle) ECOUT(handle, EINFO)
#define ECOUT_D(handle) ECOUT(handle, EDEBUG)
#define ECOUT_H(handle) ECOUT(handle, EHINTS)
#define ECOUT_T(handle) ECOUT(handle, ETRACE)

/**
  * @brief macro of stream message output without elog instance
  * @param[in] lvl      log level
  */
#define FCOUT(lvl) ECOUT(1, lvl)

/**
  * @brief macro of stream message output
  * @param[in] handle   module handle @see elog_base
  * @param[in] lvl      log level
  */
#define UCOUT(handle, lvl) \
  if((handle)->is_on(lvl)) \
    elog::emessage(handle, __FILE__, __LINE__, lvl).stream()

/**
  * @brief macro of printf message output
  * @param[in] handle   module handle return from register @see register_module
  * @param[in] lvl      log level
  * @param[in] fmt      printf-like output format definition
  * @param[in] ...      variable parameter
  */
#define EPRINTF(handle, lvl, fmt, ...) \
  if(elog::module_is_on(handle, lvl)) { \
    elog::eprintf(handle, __FILE__, __LINE__, lvl, fmt, ##__VA_ARGS__); \
  }

#define EPRINTF_F(handle, fmt, ...) EPRINTF(handle, EFATAL, fmt, ##__VA_ARGS__)
#define EPRINTF_E(handle, fmt, ...) EPRINTF(handle, EERR, fmt, ##__VA_ARGS__)
#define EPRINTF_W(handle, fmt, ...) EPRINTF(handle, EWARN, fmt, ##__VA_ARGS__)
#define EPRINTF_I(handle, fmt, ...) EPRINTF(handle, EINFO, fmt, ##__VA_ARGS__)
#define EPRINTF_D(handle, fmt, ...) EPRINTF(handle, EDEBUG, fmt, ##__VA_ARGS__)
#define EPRINTF_H(handle, fmt, ...) EPRINTF(handle, EHINTS, fmt, ##__VA_ARGS__)
#define EPRINTF_T(handle, fmt, ...) EPRINTF(handle, ETRACE, fmt, ##__VA_ARGS__)

/**
  * @brief macro of large message (>32 kB) RAW ouput with zero copy output
  * @param[in] handle   module handle return from register @see register_module
  * @param[in] lvl      log level
  * @param[in] data     message to output
  * @param[in] size     size of message
  * @param[in] fmt      pintf-link format
  * @param[in] ...      variable paramter
  */
#define EDUMP(handle, lvl, data, size, fmt, ...) \
  if(elog::module_is_on(handle, lvl)) { \
    elog::edump(handle, __FILE__, __LINE__, lvl, data, size, fmt, ##__VA_ARGS__); \
  }

/**
  * @brief macro of backtrace with out pattern
  */
#define EBACKTRACE() elog::ebacktrace()
/**
  * @brief macro of backtrace with out pattern of user message
  * @param[in] handle   module handle return from register @see register_module
  * @param[in] lvl      log level
  * @param[in] fmt      pintf-link format
  * @param[in] ...      variable paramter
  */
#define EBACKTRACE_PRINTF(handle, lvl, fmt, ...) \
  if(elog::module_is_on(handle, lvl)) \
    elog::ebacktrace(handle, __FILE__, __LINE__, lvl, fmt, ##__VA_ARGS__)

/**
  * @brief macro of condition log, output when condition and lvl is true
  * @param[in] handle     module handle return from register @see register_module
  * @param[in] lvl        log level
  * @param[in] condition  condition formula, e.g. "10 < 20" is true
  */
#define EIF(handle, lvl, condition) \
  if((condition) && elog::module_is_on(handle, lvl))             \
    elog::emessage(handle, __FILE__, __LINE__, lvl).stream()


/**
  * @brief macro of assert when condition is false, a fatal message output, enable for debug model.
  * @param[in] handle     module handle return from register @see register_module
  * @param[in] condition  condition formula, e.g. "10 > 20" is false
  */
#ifdef NDEBUG
#define EASSERT(handle, condition) ((void)0)
#define FASSERT(condition) ((void)0)
#else
#define EASSERT(handle, condition)  \
  EIF(handle, elog::level::fatal, !(condition)) << "Assert failed: " #condition
#define FASSERT(condition) EASSERT(1, condition)
#endif

/**
  * @brief macro of assert when condition is false without handle, a fatal message output, enable for debug model.
  * @param[in] handle     module handle return from register @see register_module
  * @param[in] condition  condition formula, e.g. "10 > 20" is false
  */
#ifdef NDEBUG
#define FASSERT(condition) ((void)0)
#else
#define FASSERT(condition) EASSERT(1, condition)
#endif

/**
  * @brief macro of check when condition is false, a fatal message output.
  * @param[in] condition  condition formula, e.g. "10 > 20" is false
  */
#define ECHECK(condition)                                       \
  if (!(condition))                                            \
    elog::efatal_message(__FILE__, __LINE__, "Check failed: " #condition " ", false).stream()

/**
  * @brief group of macro for check_xx when condition is false, a fatal message output.
  * @param[in] val1 the 1st parameter of compare formula
  * @param[in] val2 the 2nd parameter of compare formula
  */
#define ECHECK_EQ(val1, val2) ECHECK_OP(_EQ, ==, val1, val2)
#define ECHECK_NE(val1, val2) ECHECK_OP(_NE, !=, val1, val2)
#define ECHECK_LE(val1, val2) ECHECK_OP(_LE, <=, val1, val2)
#define ECHECK_LT(val1, val2) ECHECK_OP(_LT, < , val1, val2)
#define ECHECK_GE(val1, val2) ECHECK_OP(_GE, >=, val1, val2)
#define ECHECK_GT(val1, val2) ECHECK_OP(_GT, > , val1, val2)

/**
  * @brief macro of module is enable on give level
  * @param[in] handle     module handle return from register @see register_module
  * @param[in] lvl        log level
  */
#define EMODULE_IS_ON(handle, lvl) elog::module_is_on(handle, lvl)
