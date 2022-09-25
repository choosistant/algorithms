from dataclasses import dataclass, field
from typing import List


@dataclass
class AnnotatedTextSegment:
    label: str
    segment_start: int
    segment_end: int
    segment: str


@dataclass
class AnnotatedExample:
    review_text: str
    labels: List[AnnotatedTextSegment] = field(default_factory=list)


def parse_annotation_from_label_studio(exported_annotation_item) -> AnnotatedExample:
    review_text = exported_annotation_item["data"]["reviewText"]
    example = AnnotatedExample(review_text=review_text)

    for annotation_object in exported_annotation_item["annotations"]:
        labeled_segments = annotation_object["result"]
        for labeled_segment in labeled_segments:
            vals = labeled_segment["value"]
            example.labels.append(
                AnnotatedTextSegment(
                    label=vals["labels"][0],
                    segment_start=vals["start"],
                    segment_end=vals["end"],
                    segment=vals["text"],
                )
            )
    return example
