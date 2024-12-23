def retrieve_documents(state):
    prompt_type = state["prompt_type"]
    query = f"Retrieve rubric, example essays, and all relevant historical chapters for {prompt_type} prompts."
    try:
        state["documents"] = get_relevant_documents(query, prompt_type)
        logger.info(f"Retrieved {len(state['documents'])} relevant documents.")
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise RuntimeError(f"Error retrieving documents: {e}")
    return state
