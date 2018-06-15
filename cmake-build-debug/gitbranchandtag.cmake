# Branch:
   execute_process(COMMAND "/usr/local/bin/git" branch .
     RESULT_VARIABLE hadbranch ERROR_VARIABLE error
     WORKING_DIRECTORY /Users/jeasungpark/CLionProjects/Quantization Database)
   if(NOT hadbranch)
     execute_process(COMMAND "/usr/local/bin/git" push origin .
      WORKING_DIRECTORY "/Users/jeasungpark/CLionProjects/Quantization Database")
   endif()
   set(TAG_BRANCH .)

   # Create or move tag
   execute_process(
     COMMAND "/usr/local/bin/git" tag -f  
     COMMAND "/usr/local/bin/git" push --tags
     RESULT_VARIABLE notdone WORKING_DIRECTORY "/Users/jeasungpark/CLionProjects/Quantization Database")
   if(notdone)
     message(FATAL_ERROR
        "Error creating tag  on branch ")
   endif()