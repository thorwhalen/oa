"""
Parameters, configuration, and constants for the OpenAI Gym environment.

Some resources:
* OpenAI API pricing: https://platform.openai.com/docs/pricing

"""

turns_data_ssot = {
    "id": {
        "description": "A unique identifier for the conversation turn.",
        "example": "1fc35aa7-6b7a-4dae-9838-ead52c6d4793",
    },
    "children": {
        "description": "An array of child conversation turns, which can hold additional messages in the conversation thread.",
        "example": "[]",
    },
    "message.id": {
        "description": "A unique identifier for the message within the conversation turn.",
        "example": "1fc35aa7-6b7a-4dae-9838-ead52c6d4793",
    },
    "message.author.role": {
        "description": "The role of the author of the message (e.g., user, assistant).",
        "example": "assistant",
    },
    "message.author.metadata.real_author": {
        "description": "Metadata indicating the real author or source of the message.",
        "example": "tool:web",
    },
    "message.author.name": {
        "description": "The name of the author of the message.",
        "example": "dalle.text2im",
    },
    "message.content.content_type": {
        "description": "The type of content in the message (e.g., text, image, etc.).",
        "example": "text",
    },
    "message.content.parts": {
        "description": "An array containing parts or segments of the message content, typically for handling long messages.",
        "example": [
            "The generated image visually ca...lements in a futuristic design."
        ],
    },
    "message.content.model_set_context": {
        "description": "Context information related to the model used for content generation.",
        "example": "",
    },
    "message.content.language": {
        "description": "The language of the message content, represented in standard language codes.",
        "example": "unknown",
    },
    "message.content.text": {
        "description": "The actual text of the message.",
        "example": 'search("Please give me an estim...rian diet and a omnivore diet")',
    },
    "message.status": {
        "description": "The processing status of the message (e.g., finished, in-progress).",
        "example": "finished_successfully",
    },
    "message.end_turn": {
        "description": "A boolean indicating if this is the last message in the conversation turn.",
        "example": True,
    },
    "message.weight": {
        "description": "A numeric value representing the message's importance or relevance in the conversation.",
        "example": 1,
    },
    "message.metadata.is_visually_hidden_from_conversation": {
        "description": "A boolean indicating if the message is hidden from the visible conversation stream.",
        "example": True,
    },
    "message.metadata.shared_conversation_id": {
        "description": "An identifier for the shared context of the conversation, if applicable.",
        "example": "678a1339-d14c-8013-bfcb-288d367a9079",
    },
    "message.metadata.user_context_message_data": {
        "description": "Contextual data related to the user's message, if applicable.",
        "example": None,
    },
    "message.metadata.is_user_system_message": {
        "description": "A boolean indicating if the message is generated as a system message for the user.",
        "example": True,
    },
    "message.metadata.is_redacted": {
        "description": "A boolean indicating if the message content has been redacted for privacy or security reasons.",
        "example": True,
    },
    "message.metadata.request_id": {
        "description": "An identifier for the request associated with the message, useful for debugging.",
        "example": "9034eeef6e62e209-MRS",
    },
    "message.metadata.message_source": {
        "description": "Information about the source of the message, if applicable.",
        "example": None,
    },
    "message.metadata.timestamp_": {
        "description": "The timestamp format type for message creation, indicating if it's absolute or relative.",
        "example": "absolute",
    },
    "message.metadata.message_type": {
        "description": "The type/category of message, useful for filtering or processing messages.",
        "example": None,
    },
    "message.metadata.model_slug": {
        "description": "A slug representing the model used to generate the response.",
        "example": "gpt-4o",
    },
    "message.metadata.default_model_slug": {
        "description": "The default model slug for the message, representing the standard model used.",
        "example": "gpt-4o",
    },
    "message.metadata.parent_id": {
        "description": "The ID of the parent message for threading purposes.",
        "example": "073e2336-5c95-434e-a0d2-74a58b68f8e0",
    },
    "message.metadata.finish_details.type": {
        "description": "The type of finish that occurred for the message processing (e.g., stop, timeout).",
        "example": "stop",
    },
    "message.metadata.finish_details.stop_tokens": {
        "description": "An array of token IDs that indicate where the message generation stopped.",
        "example": [200002, 200007],
    },
    "message.metadata.is_complete": {
        "description": "A boolean indicating if the message generation process was completed successfully.",
        "example": True,
    },
    "message.metadata.citations": {
        "description": "An array of citations included in the message, if applicable.",
        "example": "[]",
    },
    "message.metadata.content_references": {
        "description": "References to additional content used in the message, if any.",
        "example": "[]",
    },
    "message.metadata.command": {
        "description": "The command issued by the user that generated this message.",
        "example": "search",
    },
    "message.metadata.status": {
        "description": "The status of the message at the time of capture (completed, in-progress, etc.).",
        "example": "finished",
    },
    "message.metadata.search_source": {
        "description": "The source from which search results were derived, if applicable.",
        "example": "composer_search",
    },
    "message.metadata.client_reported_search_source": {
        "description": "The source reported by the client regarding the search origin.",
        "example": "conversation_composer_previous_web_mode",
    },
    "message.metadata.search_result_groups": {
        "description": "An array of search result groups that provide relevant information based on the user's query.",
        "example": [
            {
                "type": "search_result_group",
                "domain": "learnmetrics.com",
                "entries": [
                    {
                        "type": "search_result",
                        "url": "https://learnmetrics.com/how-ma...average-home-electricity-usage/",
                        "title": "How Many kWh Per Day Is Normal? Average 1-6 Person Home kWh Usage",
                        "snippet": "7,340 kWh Per Year: 2 Person Ho...: 4 Person Home: 36.58 kWh P...",
                        "ref_id": {
                            "turn_index": 0,
                            "ref_type": "search",
                            "ref_index": 0,
                        },
                        "content_type": None,
                        "pub_date": None,
                        "attributions": None,
                    }
                ],
            }
        ],
    },
    "message.metadata.safe_urls": {
        "description": "An array of URLs considered safe for sharing, derived from the content.",
        "example": [
            "https://www.sciencing.com/being...ls-3342/?utm_source=chatgpt.com"
        ],
    },
    "message.metadata.message_locale": {
        "description": "The locale in which the message was generated, formatted as a language-country code.",
        "example": "en-US",
    },
    "message.metadata.image_results": {
        "description": "An array of generated image results related to the conversation, if any.",
        "example": "[]",
    },
    "message.recipient": {
        "description": "The intended recipient of the message, indicating if it was meant for a specific user or a group.",
        "example": "all",
    },
    "message.create_time": {
        "description": "The creation timestamp of the message, represented as a float for more precision.",
        "example": 1737102115.45064,
    },
    "parent": {
        "description": "The ID of the parent turn of the conversation, refers to the context or previous message.",
        "example": "073e2336-5c95-434e-a0d2-74a58b68f8e0",
    },
}


metadata_ssot = {
    "title": {
        "description": "The title of the chat conversation.",
        "example": "Test Chat 1",
    },
    "create_time": {
        "description": "A timestamp indicating when the chat conversation was created, represented in Unix time format.",
        "example": 1737020729.060687,
    },
    "update_time": {
        "description": "A timestamp indicating the last time the chat conversation was updated, represented in Unix time format.",
        "example": 1737020733.031014,
    },
    "moderation_results": {
        "description": "An array holding the results of moderation checks applied to the conversation. If no moderation has taken place, this array will be empty.",
        "example": [],
    },
    "current_node": {
        "description": "The unique identifier for the current state or node in the conversation flow, typically in UUID format.",
        "example": "be4486db-894f-4e6f-bd0a-22d9d2facf69",
    },
    "conversation_id": {
        "description": "A unique identifier for the conversation as a whole, typically in UUID format.",
        "example": "6788d539-0f2c-8013-9535-889bf344d7d5",
    },
    "is_archived": {
        "description": "A boolean indicating whether the chat conversation has been archived. True means it is archived, false means it is active.",
        "example": False,
    },
    "safe_urls": {
        "description": "An array of safe URLs that were included in the conversation. If there are no safe URLs, this array will be empty.",
        "example": [],
    },
    "default_model_slug": {
        "description": "A string representing the default model used during the conversation, which specifies the AI language model employed.",
        "example": "gpt-4o",
    },
    "disabled_tool_ids": {
        "description": "An array containing the identifiers of any tools that have been disabled during the conversation. If none are disabled, this array will be empty.",
        "example": [],
    },
    "is_public": {
        "description": "A boolean indicating whether the conversation is accessible to the public. True means it is public, false means it is private.",
        "example": True,
    },
    "has_user_editable_context": {
        "description": "A boolean indicating whether the user can modify the context of the conversation. True means editable, false means not editable.",
        "example": False,
    },
    "continue_conversation_url": {
        "description": "A URL that allows users to continue the conversation from a specific point. This link will redirect to the conversation session.",
        "example": "https://chatgpt.com/share/6788d539-0f2c-8013-9535-889bf344d7d5/continue",
    },
    "moderation_state": {
        "description": "An object holding the state of moderation checks applied to the conversation, providing details on whether different moderation actions have taken place.",
        "example": {
            "has_been_moderated": False,
            "has_been_blocked": False,
            "has_been_accepted": False,
            "has_been_auto_blocked": False,
            "has_been_auto_moderated": False,
        },
    },
    "is_indexable": {
        "description": "A boolean indicating whether the chat conversation can be indexed for search purposes. True means it is indexable, false means it is not.",
        "example": False,
    },
    "is_better_metatags_enabled": {
        "description": "A boolean indicating whether enhanced metatags are enabled for the conversation. True means better metatags are enabled, implying improved discoverability, while false means they are not.",
        "example": True,
    },
}
