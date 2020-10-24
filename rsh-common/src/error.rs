use crate::src::Span;

#[derive(Debug)]
pub struct Error {
    msg: String,
    spans: Vec<Span>,
    hints: Vec<String>,
}

impl Error {
    pub fn custom(msg: String) -> Self {
        Self {
            msg,
            spans: Vec::new(),
            hints: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }

    pub fn with_hint(mut self, hint: String) -> Self {
        self.hints.push(hint);
        self
    }
}

#[cfg(feature = "codespan-reporting")]
mod codespan {
    use super::Error;
    use codespan_reporting::diagnostic::{Diagnostic, Label};

    impl Error {
        pub fn codespan_diagnostic<FileId: Copy>(self, file_id: FileId) -> Diagnostic<FileId> {
            Diagnostic::error()
                .with_message(self.msg)
                .with_labels(
                    self.spans
                        .into_iter()
                        .filter_map(|span| span.as_range())
                        .map(|span| Label::primary(file_id, span))
                        .collect(),
                )
                .with_notes(self.hints)
        }
    }
}
