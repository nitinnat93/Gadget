/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class jnipegasos_JNIPegasosInterface */

#ifndef _Included_jnipegasos_JNIPegasosInterface
#define _Included_jnipegasos_JNIPegasosInterface
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     jnipegasos_JNIPegasosInterface
 * Method:    trainmodel
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljnipegasos/LearningParameter;)V
 */
JNIEXPORT void JNICALL Java_jnipegasos_JNIPegasosInterface_trainmodel
  (JNIEnv *, jobject, jstring, jstring, jstring, jobject);

/*
 * Class:     jnipegasos_JNIPegasosInterface
 * Method:    classify
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_jnipegasos_JNIPegasosInterface_classify
  (JNIEnv *, jobject, jstring, jstring, jstring);

#ifdef __cplusplus
}
#endif
#endif
