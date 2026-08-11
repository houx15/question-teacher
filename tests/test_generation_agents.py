from app.generation import LessonGenerationService

def test_generation_service_has_no_legacy_whole_lesson_loop_methods():
    obsolete = {
        "_create_narrative",
        "_create_validated_narrative",
        "_create_materials",
        "_create_validated_materials",
        "_compose_draft",
        "_review",
        "_revise",
        "_validate_narrative",
        "_validate_draft",
    }

    assert obsolete.isdisjoint(vars(LessonGenerationService))
    assert not hasattr(LessonGenerationService, "MAX_REVISIONS")
