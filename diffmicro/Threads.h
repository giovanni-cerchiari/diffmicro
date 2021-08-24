// author : Ruggero Caravita, Bartezzaghi Andrea


#ifndef _H_THREADS
#define _H_THREADS

#define USE_FAKE_THREADS
#define	FASTLIB_API	

#ifdef USE_PTHREAD
#include <pthread.h>
//#ifdef _DEBUG
//#pragma comment(lib,"pthreadVC2d.lib")
//#else
#pragma comment(lib,"pthreadVC2.lib")
//#endif
typedef pthread_t sThreadObj;
typedef pthread_mutex_t sThreadMutexObj;
#elif defined(USE_FAKE_THREADS)
	// dummy threads.. only for debugging!
#include <windows.h>
typedef HANDLE sThreadObj;
typedef CRITICAL_SECTION sThreadMutexObj;
#else
#error Thread type not specified in Config.h
#endif

enum {
	THREAD_STOPPED = 0,
	THREAD_RUNNING
};

class FASTLIB_API cThread {
public:
	cThread();
	cThread(void* (*EntryFunction)(void*));
	bool Start(void* Args);
	bool Stop(bool Force = false);
	void WaitUntilExit();
	void* GetExitValue() const;
	void Attach(void* (*EntryFunction)(void*));
	void Detach();
	bool Running();
protected:
	typedef struct sThreadArgs {
		void* CallingThread;
		void* Args;
	} sThreadArgs;

	sThreadArgs m_Args;
	int m_State;
	void* (*m_EntryFunction)(void*);
	void* m_ExitValue;

#if defined(USE_PTHREAD)
	pthread_t m_Thread;
#elif defined(USE_FAKE_THREADS)
	HANDLE m_Thread;
	HANDLE m_TerminateEvent; // event that is asserted when the thread should exit
#endif

	void Exit(void* ExitValue);
	void DoEvents();

#if defined(USE_PTHREAD)
	static void* EntryPoint(void* ThreadArgs);
#elif defined(USE_FAKE_THREADS)
#ifdef _M_CEE_PURE
	static DWORD __clrcall EntryPoint(LPVOID ThreadArgs);
#else
	static DWORD WINAPI EntryPoint(LPVOID ThreadArgs);
#endif
#endif

	virtual void* ThreadFunction(void* Args);
	virtual void ThreadCtor();
	virtual void ThreadDtor();

	bool ShouldExit(); // check if the thread should exit
};

template <class T> class ReferenceLocker;

template <typename T>
class Mutex {
protected:
	sThreadMutexObj m_Mutex;
	T m_Value;
	Mutex(const Mutex<T>& Copy) { } // lock copy constructor
	Mutex<T>& operator=(Mutex<T>& Value); // lock mutex copy
	bool m_Locked;
public:
	Mutex();
	Mutex(const T& Value);
	~Mutex();

	void Lock();
	void Unlock();
	// Default explicit construct
	T& GetRef();
	// This makes sense whether T is pointer type or not
	T& operator*();
	// This makes sense ONLY if T is a pointer type
	ReferenceLocker<T> operator->();

	operator const T(); // l-value
	T& operator=(T& Value); // writable r-value
	const T& operator=(const T& Value); // assignment
};

template <class T>
class ReferenceLocker {
protected:
	Mutex<T>* m_Ref;
public:
	ReferenceLocker(Mutex<T>& Mutex) {
		m_Ref = &Mutex;
		m_Ref->Lock();
	}

	~ReferenceLocker() {
		m_Ref->Unlock();
	}

	T& operator->() {
		return m_Ref->GetRef();
	}

	/*operator const T() {
		return m_Value;
	}*/
};

template <typename T>
Mutex<T>::Mutex() {
#ifdef USE_PTHREAD
	pthread_mutexattr_t MutexAttributes;
	pthread_mutexattr_init(&MutexAttributes);
	pthread_mutexattr_settype(&MutexAttributes, PTHREAD_MUTEX_RECURSIVE);
	if (pthread_mutex_init(&m_Mutex, &MutexAttributes) != 0) {
		__RaiseError("Impossible to init mutex!");
	}
	m_Locked = false;
#elif defined(USE_FAKE_THREADS)
	InitializeCriticalSection(&m_Mutex);
	m_Locked = false;
#endif
}

template <typename T>
Mutex<T>::Mutex(const T& Value) {
#ifdef USE_PTHREAD
	m_Value = Value;
	m_Locked = false;
	pthread_mutexattr_t MutexAttributes;
	pthread_mutexattr_init(&MutexAttributes);
	pthread_mutexattr_settype(&MutexAttributes, PTHREAD_MUTEX_RECURSIVE);
	if (pthread_mutex_init(&m_Mutex, &MutexAttributes) != 0) {
		__RaiseError("Impossible to init mutex!");
	}
#elif defined(USE_FAKE_THREADS)
	InitializeCriticalSection(&m_Mutex);
	m_Value = Value;
	m_Locked = false;
#endif
}

template <typename T>
Mutex<T>::~Mutex() {
	Lock();
#ifdef USE_PTHREAD
	pthread_mutex_destroy(&m_Mutex);
#elif defined(USE_FAKE_THREADS)
	DeleteCriticalSection(&m_Mutex);
#endif
}

template <typename T>
inline void Mutex<T>::Lock() {
#ifdef USE_PTHREAD
	int Ret = pthread_mutex_lock(&m_Mutex);
	if (Ret != 0) {
		BREAKPOINT;
		if (Ret == EBUSY) {
			while (pthread_mutex_lock(&m_Mutex) == EBUSY);
		}
	}
	m_Locked = true;
#elif defined(USE_FAKE_THREADS)
	EnterCriticalSection(&m_Mutex);
	m_Locked = true;
#endif
}

template <typename T>
inline void Mutex<T>::Unlock() {
#ifdef USE_PTHREAD
	m_Locked = false;
	int Ret = pthread_mutex_unlock(&m_Mutex);
	if (Ret != 0) {
		BREAKPOINT;
	}
#elif defined(USE_FAKE_TRHEADS)
	m_Locked = false;
	LeaveCriticalSection(&m_Mutex);
#endif
}

/*template <typename T>
inline T& Mutex<T>::GetRef() {
	if (!m_Locked) {
		__RaiseError("Mutex not locked!");
	}
	return m_Value;
}*/

template <typename T>
inline T& Mutex<T>::operator*() {
	return this->GetRef();
}

template <typename T>
ReferenceLocker<T> Mutex<T>::operator->() {
	return ReferenceLocker<T>(*this);
}

template <typename T>
inline Mutex<T>::operator const T() {
	Lock();
	T Return = m_Value;
	Unlock();
	return Return;
}

template <typename T>
inline T& Mutex<T>::operator=(T& Value) {
	Lock();
	m_Value = Value;
	Unlock();
	return Value;
}

template <typename T>
inline const T& Mutex<T>::operator=(const T& Value) {
	Lock();
	m_Value = Value;
	Unlock();
	return Value;
}

#endif // _H_THREADS
