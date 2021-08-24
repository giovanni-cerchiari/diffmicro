
// author : Ruggero Caravita, Bartezzaghi Andrea

#include "stdafx.h"
#include <iostream>

#include "Threads.h"

cThread::cThread() {
	m_EntryFunction = NULL;
	m_ExitValue = NULL;
	m_State = THREAD_STOPPED;
}

cThread::cThread(void* (*EntryFunction)(void*)) {
	cThread();
	m_EntryFunction = EntryFunction;
}

bool cThread::Start(void* Args) {
#if defined(USE_PTHREAD)
	// build args
	m_Args.CallingThread = this;
	m_Args.Args = Args;
	// create the thread
	if (pthread_create(&m_Thread, NULL, EntryPoint, (void*)&m_Args) != 0) {
		std::cerr << "Error creating cThread" << std::endl;
		return false;
	}
#elif defined(USE_FAKE_THREADS)
	// create the terminate event
	m_TerminateEvent = CreateEvent(NULL, true, false, NULL);
	if (m_TerminateEvent == NULL) {
		std::cerr << "Error creating terminate event" << std::endl;
		return false;
	}
	// build args
	m_Args.CallingThread = this;
	m_Args.Args = Args;
	// create the thread
	m_Thread = CreateThread(NULL, 0, EntryPoint, (void*)&m_Args, 0, NULL);
	if (m_Thread == NULL) {
		std::cerr << "Error starting thread" << std::endl;
		return false;
	}
#endif
	m_State = THREAD_RUNNING;
	return true;
}

// check if the thread should exit
bool cThread::ShouldExit() {
#if defined(USE_PTHREAD)
	// TODO!
#error Not implemented yet!
#elif defined(USE_FAKE_THREADS)
	// check if the terminate event has been signaled
	if (WaitForSingleObject(m_TerminateEvent, 0) == WAIT_OBJECT_0)
		return true;
	else
		return false;
#endif
}

bool cThread::Stop(bool Force) {
#if defined(USE_PTHREAD)
	pthread_kill(m_Thread, 0);
	m_State = THREAD_STOPPED;
	return true;
#elif defined(USE_FAKE_THREADS)
	// signal the terminate event
	SetEvent(m_TerminateEvent);
	// TODO: wait for thread to terminate
	WaitUntilExit();
	m_State = THREAD_STOPPED;
	return true;
#endif
}

void cThread::WaitUntilExit() {
	// easy situation?
	if (m_State != THREAD_RUNNING) {
#ifdef USE_FAKE_THREADS
		CloseHandle(m_TerminateEvent);
		CloseHandle(m_Thread);
#endif
		return;
	}
#if defined(USE_PTHREAD)
	pthread_join(m_Thread, &m_ExitValue);
#elif defined(USE_FAKE_THREADS)
	// wait for it to exit
	if (WaitForSingleObject(m_Thread, INFINITE) == WAIT_OBJECT_0) {
		// get exit value
		//GetExitCodeThread(m_Thread,&m_ExitValue); TODO!
	}
	CloseHandle(m_TerminateEvent);
	CloseHandle(m_Thread);
#endif
}

void* cThread::GetExitValue() const {
	return m_ExitValue;
}

void cThread::Attach(void* (*EntryFunction)(void*)) {
	if (m_State != THREAD_STOPPED) {
		std::cerr << "Warning: entry function could not be set while running" << std::endl;
		return;
	}
	m_EntryFunction = EntryFunction;
}

void cThread::Detach() {
	if (m_State != THREAD_STOPPED) {
		std::cerr << "Warning: entry function could not be unset while running" << std::endl;
		return;
	}
	m_EntryFunction = NULL;
}

bool cThread::Running() {
	if (m_State == THREAD_RUNNING)
		return true;
	else
		return false;
}

void cThread::Exit(void* ExitValue) {
	if (m_State == THREAD_RUNNING) {
		m_ExitValue = ExitValue;
		m_State = THREAD_STOPPED;
#if defined(USE_PTHREAD)
		pthread_exit(ExitValue);
#elif defined(USE_FAKE_THREADS)
		// exit
		ExitThread(0);
#endif
	}
}

void cThread::DoEvents() {
	Sleep(0);
}

#if defined(USE_PTHREAD)
void* cThread::EntryPoint(void* ThreadArgs) {
#elif defined(USE_FAKE_THREADS)
#ifdef _M_CEE_PURE
DWORD __clrcall cThread::EntryPoint(LPVOID ThreadArgs) {
#else
DWORD WINAPI cThread::EntryPoint(LPVOID ThreadArgs) {
#endif
#endif
	// get args
	sThreadArgs* Args = (sThreadArgs*)ThreadArgs;
	cThread* Parent = (cThread*)Args->CallingThread;
	// use an external function?
	if (Parent->m_EntryFunction) {
		Parent->m_ExitValue = (*Parent->m_EntryFunction)(Args->Args);
		Parent->DoEvents();
	}
	else {
		// call ctor
		Parent->ThreadCtor();
		Parent->DoEvents();
		// call function
		Parent->m_ExitValue = Parent->ThreadFunction(Args->Args);
		// call dtor
		Parent->DoEvents();
		Parent->ThreadDtor();
	}
	Parent->m_State = THREAD_STOPPED;

#if defined(USE_PTHREAD)
	return Parent->m_ExitValue;
#elif defined(USE_FAKE_THREADS)
	return 0;
#endif
}

void* cThread::ThreadFunction(void* Args) {
	Sleep(0);
	return Args;
}

void cThread::ThreadCtor() {

}

void cThread::ThreadDtor() {

}
