# src/run_communicator_agent_test.py
"""
Test runner for Enhanced COMMUNICATOR Agent v1.0
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.communicator_enhanced import (
    EnhancedCommunicatorAgent,
    get_communicator_agent,
    RecipientManager,
    TemplateLibrary,
    ChannelSelector,
    NotificationBatcher,
    MessageType,
    MessagePriority,
    Channel,
    RecipientRole,
    DeliveryStatus
)


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def test_recipient_manager():
    """Test recipient management."""
    print_separator("TEST 1: Recipient Manager")
    
    manager = RecipientManager()
    
    print(f"\nTotal Profiles: {len(manager.profiles)}")
    
    # Test get by ID
    print("\nGet by ID (cra_001):")
    profile = manager.get_profile("cra_001")
    if profile:
        print(f"  Name: {profile.name}")
        print(f"  Email: {profile.email}")
        print(f"  Role: {profile.role.value}")
        print(f"  Preferred Channel: {profile.preferred_channel.value}")
        print(f"  Sites: {profile.sites}")
    
    # Test get by role
    print("\nGet by Role (CRA):")
    cras = manager.get_profiles_by_role(RecipientRole.CRA)
    print(f"  Found: {len(cras)}")
    for cra in cras:
        print(f"    - {cra.name}")
    
    # Test get by site
    print("\nGet by Site (Site_1):")
    site_profiles = manager.get_profiles_for_site("Site_1")
    print(f"  Found: {len(site_profiles)}")
    for p in site_profiles:
        print(f"    - {p.name} ({p.role.value})")
    
    return len(manager.profiles) > 0


def test_template_library():
    """Test message templates."""
    print_separator("TEST 2: Template Library")
    
    library = TemplateLibrary()
    
    print(f"\nTotal Templates: {len(library.templates)}")
    
    # List templates
    print("\nAvailable Templates:")
    for template_id, template in library.templates.items():
        print(f"  - {template_id}: {template.name} ({template.message_type.value})")
    
    # Test template rendering
    print("\nRendering 'query_reminder' template:")
    template = library.get_template("query_reminder")
    
    if template:
        context = {
            "recipient_name": "Dr. Smith",
            "query_count": 15,
            "site_id": "Site_1",
            "aged_queries": 5,
            "recent_queries": 10
        }
        
        subject, body = template.render(context)
        print(f"  Subject: {subject}")
        print(f"  Body Preview: {body[:100]}...")
    
    # Test get by type
    print("\nTemplates by Type (REMINDER):")
    reminders = library.get_templates_by_type(MessageType.REMINDER)
    print(f"  Found: {len(reminders)}")
    
    return len(library.templates) > 0


def test_channel_selector():
    """Test channel selection."""
    print_separator("TEST 3: Channel Selector")
    
    selector = ChannelSelector()
    manager = RecipientManager()
    
    # Get a test profile
    profile = manager.get_profile("cra_001")
    
    print("\nChannel Selection Tests:")
    
    test_cases = [
        (MessageType.ALERT, MessagePriority.URGENT),
        (MessageType.ALERT, MessagePriority.HIGH),
        (MessageType.REMINDER, MessagePriority.NORMAL),
        (MessageType.REPORT, MessagePriority.LOW),
        (MessageType.NOTIFICATION, MessagePriority.NORMAL)
    ]
    
    for msg_type, priority in test_cases:
        channels = selector.select_channels(msg_type, priority, profile)
        print(f"  {msg_type.value} + {priority.value}: {[c.value for c in channels]}")
    
    return True


def test_notification_batcher():
    """Test notification batching."""
    print_separator("TEST 4: Notification Batcher")
    
    batcher = NotificationBatcher()
    manager = RecipientManager()
    
    profile = manager.get_profile("dm_001")  # DM prefers batching
    
    print(f"\nRecipient: {profile.name}")
    print(f"Batch Preference: {profile.batch_notifications}")
    
    # Create test messages
    from src.agents.communicator_enhanced import Message
    
    messages = [
        Message(
            message_id=f"MSG-TEST-{i}",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.LOW,
            subject=f"Test Notification {i}",
            body=f"Test body {i}",
            sender="system",
            recipients=[profile],
            channels=[Channel.EMAIL]
        )
        for i in range(5)
    ]
    
    print("\nAdding messages to batch:")
    for msg in messages:
        if batcher.should_batch(msg, profile):
            batcher.add_to_batch(msg, profile)
            print(f"  ✓ {msg.message_id} added to batch")
        else:
            print(f"  ✗ {msg.message_id} not batched")
    
    print(f"\nQueue Size: {batcher.get_queue_size(profile.recipient_id)}")
    
    # Create batch
    print("\nCreating batch:")
    batch = batcher.create_batch(profile, "daily_digest")
    
    print(f"  Batch ID: {batch.batch_id}")
    print(f"  Type: {batch.batch_type}")
    print(f"  Messages: {batch.total_messages}")
    print(f"  Status: {batch.status.value}")
    
    return batch.total_messages == 5


def test_message_drafting():
    """Test message drafting."""
    print_separator("TEST 5: Message Drafting")
    
    agent = get_communicator_agent()
    
    # Draft from template
    print("\nDrafting from template:")
    
    context = {
        "query_count": 20,
        "site_id": "Site_1",
        "aged_queries": 8,
        "recent_queries": 12
    }
    
    try:
        message = agent.draft_message(
            template_id="query_reminder",
            recipients=["site_001"],
            context=context
        )
        
        print(f"  Message ID: {message.message_id}")
        print(f"  Type: {message.message_type.value}")
        print(f"  Priority: {message.priority.value}")
        print(f"  Subject: {message.subject}")
        print(f"  Recipients: {[r.name for r in message.recipients]}")
        print(f"  Channels: {[c.value for c in message.channels]}")
        print(f"  Status: {message.status.value}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Draft custom message
    print("\nDrafting custom message:")
    
    try:
        custom_msg = agent.draft_custom_message(
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.NORMAL,
            subject="Custom Test Message",
            body="This is a custom test message body.",
            recipients=["cra_001"]
        )
        
        print(f"  Message ID: {custom_msg.message_id}")
        print(f"  Subject: {custom_msg.subject}")
        print(f"  Status: {custom_msg.status.value}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    return True


def test_message_workflow():
    """Test full message workflow."""
    print_separator("TEST 6: Message Workflow")
    
    agent = get_communicator_agent()
    
    # Draft message
    print("\n1. Drafting message:")
    message = agent.draft_custom_message(
        message_type=MessageType.NOTIFICATION,
        priority=MessagePriority.HIGH,
        subject="Workflow Test",
        body="Testing the full message workflow.",
        recipients=["cra_001"]
    )
    print(f"   Status: {message.status.value}")
    
    # Queue message
    print("\n2. Queueing message:")
    queue_result = agent.queue_message(message)
    print(f"   Result: {queue_result}")
    print(f"   Status: {message.status.value}")
    
    # Send message
    print("\n3. Sending message:")
    send_result = agent.send_message(message)
    print(f"   Sent: {send_result}")
    print(f"   Status: {message.status.value}")
    print(f"   Sent Time: {message.sent_time}")
    print(f"   Delivered Time: {message.delivered_time}")
    
    return message.status == DeliveryStatus.DELIVERED


def test_approval_workflow():
    """Test message approval workflow."""
    print_separator("TEST 7: Approval Workflow")
    
    agent = get_communicator_agent()
    
    # Draft an escalation (requires approval)
    print("\n1. Drafting escalation (requires approval):")
    
    context = {
        "site_id": "Site_1",
        "issue_type": "Open Queries",
        "escalation_reason": "No response",
        "days_outstanding": 21,
        "impact_description": "High impact",
        "previous_actions": "Multiple reminders",
        "required_action": "Immediate review"
    }
    
    message = agent.draft_message(
        template_id="escalation",
        recipients=["lead_001"],
        context=context
    )
    
    print(f"   Requires Approval: {message.requires_approval}")
    
    # Queue (should go to pending approval)
    print("\n2. Queueing message:")
    queue_result = agent.queue_message(message)
    print(f"   Result: {queue_result}")
    print(f"   Status: {message.status.value}")
    
    # Get pending approvals
    print("\n3. Pending approvals:")
    pending = agent.get_pending_approvals()
    print(f"   Count: {len(pending)}")
    
    # Approve
    print("\n4. Approving message:")
    approved = agent.approve_message(message.message_id, "manager_001")
    print(f"   Approved: {approved}")
    print(f"   Status: {message.status.value}")
    print(f"   Approved By: {message.approved_by}")
    
    # Now can send
    print("\n5. Sending approved message:")
    send_result = agent.send_message(message)
    print(f"   Sent: {send_result}")
    print(f"   Final Status: {message.status.value}")
    
    return message.status == DeliveryStatus.DELIVERED


def test_role_based_messaging():
    """Test role-based message distribution."""
    print_separator("TEST 8: Role-Based Messaging")
    
    agent = get_communicator_agent()
    
    context = {
        "signature_count": 10,
        "site_id": "Site_1"
    }
    
    print("\nSending signature reminders to all CRAs:")
    
    messages = agent.draft_for_role("signature_reminder", RecipientRole.CRA, context)
    
    print(f"  Messages Created: {len(messages)}")
    
    for msg in messages:
        print(f"    - To: {msg.recipients[0].name} | Status: {msg.status.value}")
    
    print("\nSending site-specific reminders:")
    
    site_messages = agent.draft_for_site("query_reminder", "Site_1", {
        "query_count": 25,
        "aged_queries": 10,
        "recent_queries": 15
    })
    
    print(f"  Messages Created: {len(site_messages)}")
    
    return len(messages) > 0


def test_natural_language_queries():
    """Test natural language query handling."""
    print_separator("TEST 9: Natural Language Queries")
    
    agent = get_communicator_agent()
    
    queries = [
        "Send query reminder to Site_1",
        "Draft signature reminder",
        "Send SAE alert",
        "Create escalation for Site_2",
        "Create daily digest for CRA team",
        "Show message status",
        "Show pending approvals"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        start = time.time()
        result = agent.communicate_from_query(query)
        duration = time.time() - start
        
        print(f"Duration: {duration:.2f}s")
        print(f"Messages Drafted: {result.messages_drafted}")
        print(f"Messages Sent: {result.messages_sent}")
        print(f"Pending Approval: {result.messages_pending_approval}")
        print(f"Batches: {result.batches_created}")
        print(f"Summary: {result.summary[:60]}..." if len(result.summary) > 60 else f"Summary: {result.summary}")
    
    return True


def test_message_statistics():
    """Test message statistics."""
    print_separator("TEST 10: Message Statistics")
    
    agent = get_communicator_agent()
    
    # Create some messages first
    for i in range(5):
        agent.draft_custom_message(
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.NORMAL,
            subject=f"Test {i}",
            body=f"Body {i}",
            recipients=["cra_001"]
        )
    
    # Send some
    for msg in list(agent.message_history.values())[:3]:
        agent.queue_message(msg)
        agent.send_message(msg)
    
    print("\nMessage Statistics:")
    stats = agent.get_message_stats()
    
    print(f"  Total Messages: {stats['total']}")
    print(f"\n  By Status:")
    for status, count in stats['by_status'].items():
        print(f"    {status}: {count}")
    
    print(f"\n  By Type:")
    for msg_type, count in stats['by_type'].items():
        print(f"    {msg_type}: {count}")
    
    print(f"\n  By Priority:")
    for priority, count in stats['by_priority'].items():
        print(f"    {priority}: {count}")
    
    return stats['total'] > 0


def run_all_tests():
    """Run all communicator agent tests."""
    print("\n" + "=" * 70)
    print(" ENHANCED COMMUNICATOR AGENT v1.0 - TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Recipient Manager
    try:
        results['recipient_manager'] = test_recipient_manager()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['recipient_manager'] = False
    
    # Test 2: Template Library
    try:
        results['template_library'] = test_template_library()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['template_library'] = False
    
    # Test 3: Channel Selector
    try:
        results['channel_selector'] = test_channel_selector()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['channel_selector'] = False
    
    # Test 4: Notification Batcher
    try:
        results['notification_batcher'] = test_notification_batcher()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['notification_batcher'] = False
    
    # Test 5: Message Drafting
    try:
        results['message_drafting'] = test_message_drafting()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['message_drafting'] = False
    
    # Test 6: Message Workflow
    try:
        results['message_workflow'] = test_message_workflow()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['message_workflow'] = False
    
    # Test 7: Approval Workflow
    try:
        results['approval_workflow'] = test_approval_workflow()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['approval_workflow'] = False
    
    # Test 8: Role-Based Messaging
    try:
        results['role_based'] = test_role_based_messaging()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['role_based'] = False
    
    # Test 9: Natural Language Queries
    try:
        results['nl_queries'] = test_natural_language_queries()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['nl_queries'] = False
    
    # Test 10: Message Statistics
    try:
        results['statistics'] = test_message_statistics()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['statistics'] = False
    
    # Summary
    print_separator("TEST RESULTS SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASSED" if passed_flag else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)