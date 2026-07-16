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

func TestManagerPublishHonorsSuppressedContext(t *testing.T) {
	manager := NewManager(1)
	manager.Publish(WithPublishingSuppressed(context.Background()), Record{})
	manager.mu.Lock()
	defer manager.mu.Unlock()
	if len(manager.queue) != 0 {
		t.Fatalf("suppressed queue length=%d", len(manager.queue))
	}
}
