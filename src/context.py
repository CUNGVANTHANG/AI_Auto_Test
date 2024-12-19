DEFAULT_CONTEXT = """
**Task**: Analyze the provided SRS segment to extract test case requirements. Each segment may have multiple or no requirements. Include information only if at least 70% confident.
 
**Fields to Extract**:
 
**REQUIREMENT_ID**:
   Unique ID for each test case, either context-derived or generated.
   Examples: "2A_GEN_SET_00010", "4A_ECO_PUR_00290"
 
**TEST_OBJECTIVE**:
   Functionality Description under test.
   Often follows section titles or phrases like "The function will…".
   Examples: "The radio signal must be uninterrupted."
 
**PRECONDITION**:
   Conditions Required for the test.
   Look for "When," "If," "Before," or setup details.
   Examples: "When the main tuner is set to FM/AM…", "System supports dual tuner."
 
**PROCEDURE**:
   Actions performed in the test, often in command form (e.g., "Select," "Press").
   Examples: "Tune to a valid FM station.", "Set main tuner to FM."
 
**EXPECTED_RESULT**:
   Outcome expected post-procedure.
   Look for keywords like "should," "must," "will."
   Examples: "Radio signal must be seamless.", "Main tuner should play FM/AM."
 
**MISCELLANEOUS**: Additional relevant info.
 
**Format**: Output as JSON list with fields:
   REQUIREMENT_ID, TEST_OBJECTIVE, PRECONDITION, PROCEDURE, EXPECTED_RESULT, MISCELLANEOUS.
   Do not give anything beside the JSON list.

"""