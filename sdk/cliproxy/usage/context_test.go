package usage

import (
	"context"
	"testing"
)

func TestPublishingSuppressedContext(t *testing.T) {
	if PublishingSuppressed(nil) {
		t.Fatal("nil context unexpectedly suppresses usage")
	}
	if PublishingSuppressed(context.Background()) {
		t.Fatal("background context unexpectedly suppresses usage")
	}
	if !PublishingSuppressed(WithPublishingSuppressed(context.Background())) {
		t.Fatal("suppressed context was not recognized")
	}
}
