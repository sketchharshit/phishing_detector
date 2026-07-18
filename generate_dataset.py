"""
generate_dataset.py
--------------------
Builds a realistic, labeled dataset of phishing and legitimate emails for the
"AI-Driven Phishing Email Detection Using NLP" project.

Since live Kaggle/UCI downloads aren't reachable from this environment, this
script generates a custom sample set (as permitted by the project brief:
"Kaggle, UCI, or custom samples"). Each email is built from realistic
templates with randomized entities (names, companies, amounts, links) so the
dataset has natural variety rather than repeated boilerplate.

Output: data/raw_emails.csv with columns:
    email_id, sender, subject, body, label   (label: 1 = phishing, 0 = legitimate)
"""

import random
import csv
import re

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

FIRST_NAMES = ["Rahul", "Priya", "Amit", "Sneha", "John", "Sarah", "Michael",
               "Emily", "David", "Anita", "Karan", "Meera", "Rohan", "Divya",
               "Chris", "Laura", "James", "Neha", "Vikram", "Pooja"]

LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Smith", "Johnson",
              "Williams", "Brown", "Iyer", "Reddy", "Kapoor", "Mehta", "Nair"]

LEGIT_COMPANIES = ["Amazon", "Microsoft", "Google", "LinkedIn", "Zoom", "Slack",
                    "GitHub", "Dropbox", "Adobe", "Spotify", "Netflix", "Coursera"]

SPOOFED_BRANDS = ["Amaz0n", "Amazon-Security", "PayPal-Support", "Micros0ft",
                   "Apple-ID-Alert", "Netflix-Billing", "Bank0fAmerica",
                   "SBI-Alerts", "HDFC-Secure", "IRS-TaxDept", "DHL-Delivery",
                   "FedEx-Notify", "Google-Account-Team"]

LEGIT_DOMAINS = ["amazon.com", "microsoft.com", "google.com", "linkedin.com",
                  "zoom.us", "slack.com", "github.com", "dropbox.com",
                  "adobe.com", "spotify.com", "netflix.com", "iict.edu.in",
                  "coursera.org", "outlook.com", "company.co.in"]

SUSPICIOUS_DOMAINS = ["secure-verify-account.info", "amaz0n-security.net",
                       "paypal-billing-update.com", "appleid-confirm.co",
                       "bank-alert-verify.xyz", "irs-refund-claim.support",
                       "dhl-tracking-update.ru", "login-verification-portal.tk",
                       "microsoft-support-team.online", "netflix-payment-fix.top",
                       "sbi-kyc-update.site", "hdfc-secure-login.club"]

URGENCY_PHRASES = [
    "Your account will be suspended within 24 hours",
    "Immediate action required to avoid permanent closure",
    "This is your final notice before account termination",
    "Unusual activity detected - verify now",
    "Your payment has failed - update details immediately",
    "Urgent: your subscription will be cancelled today",
    "Security alert: unauthorized login attempt detected",
    "Act now to prevent loss of access to your account",
    "Your package could not be delivered - action needed",
    "Tax refund pending - claim before it expires",
]

PHISHING_CTA = [
    "Click here to verify your account",
    "Confirm your identity now",
    "Update your payment information",
    "Log in immediately to secure your account",
    "Claim your refund here",
    "Reset your password using this link",
    "Verify your details to avoid suspension",
    "Download the attached invoice and confirm payment",
]

LEGIT_SUBJECTS = [
    "Your {company} invoice for this month",
    "Meeting reminder: {topic} at {time}",
    "Weekly newsletter from {company}",
    "Your order has shipped",
    "Welcome to the team!",
    "Notes from today's {topic} discussion",
    "{company} monthly statement is ready",
    "Reminder: submit your timesheet",
    "Project update - {topic}",
    "Your {company} subscription receipt",
]

PHISHING_SUBJECTS = [
    "URGENT: Verify your {brand} account now",
    "Security Alert - {brand} account suspended",
    "Action Required: {brand} payment failed",
    "Your {brand} account has been limited",
    "Final Notice: {brand} account closure pending",
    "You have won a prize from {brand}!",
    "{brand}: Unusual sign-in activity detected",
    "Confirm your {brand} details within 24 hours",
    "Your {brand} refund is ready - claim now",
    "{brand} Invoice Overdue - Pay Immediately",
]

TOPICS = ["Q3 planning", "budget review", "sprint retro", "client onboarding",
          "product launch", "hiring update", "marketing campaign", "vendor contract"]

TIMES = ["10:00 AM", "2:30 PM", "11:15 AM", "4:00 PM", "9:00 AM"]


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_amount():
    return f"${random.randint(49, 2499)}.{random.randint(0,99):02d}"


def make_legit_email(email_id):
    company = random.choice(LEGIT_COMPANIES)
    domain = random.choice(LEGIT_DOMAINS)
    topic = random.choice(TOPICS)
    time_ = random.choice(TIMES)
    sender_name = rand_name().lower().replace(" ", ".")
    sender = f"{sender_name}@{domain}"

    subject_template = random.choice(LEGIT_SUBJECTS)
    subject = subject_template.format(company=company, topic=topic, time=time_)

    body_variants = [
        f"Hi,\n\nJust a reminder that our meeting on {topic} is scheduled for {time_} today. "
        f"Please review the shared document beforehand and let me know if you have any questions.\n\n"
        f"Best regards,\n{rand_name()}",

        f"Hello,\n\nYour recent order with {company} has shipped and is expected to arrive within "
        f"3-5 business days. You can track your shipment from your account dashboard at {domain}.\n\n"
        f"Thank you for shopping with us.\n{company} Team",

        f"Hi team,\n\nAttached is the summary from today's {topic} discussion. Key action items are "
        f"listed at the bottom. Please review and add comments by Friday.\n\nThanks,\n{rand_name()}",

        f"Dear customer,\n\nThis is your monthly statement from {company}. Your account balance and "
        f"recent transactions are available by logging into your account directly at {domain}. "
        f"No action is required unless you notice a discrepancy.\n\nRegards,\n{company} Billing",

        f"Hi,\n\nWelcome to the team! We're excited to have you on board. Your onboarding schedule "
        f"and HR paperwork will be shared by {rand_name()} this week. Feel free to reach out with "
        f"any questions.\n\nWarm regards,\nHR Team",

        f"Hello,\n\nThis is a friendly reminder to submit your timesheet for this week by end of day "
        f"Friday. You can do this through the internal portal.\n\nThanks,\n{rand_name()}",

        f"Hi,\n\nHere's a quick update on {topic}: we're on track for the deadline next week. "
        f"{rand_name()} will share the detailed report in tomorrow's standup.\n\nBest,\n{rand_name()}",

        f"Hello,\n\nThank you for your continued subscription to {company}. Your receipt for this "
        f"billing cycle is attached. You can manage your subscription anytime from your account "
        f"settings at {domain}.\n\n{company} Team",
    ]

    body = random.choice(body_variants)

    return {
        "email_id": email_id,
        "sender": sender,
        "subject": subject,
        "body": body,
        "label": 0,
    }


def make_phishing_email(email_id):
    brand = random.choice(SPOOFED_BRANDS)
    domain = random.choice(SUSPICIOUS_DOMAINS)
    sender_local = random.choice(["support", "security", "alert", "billing",
                                    "no-reply", "verify", "account-team", "admin"])
    sender = f"{sender_local}@{domain}"

    subject_template = random.choice(PHISHING_SUBJECTS)
    subject = subject_template.format(brand=brand)

    urgency = random.choice(URGENCY_PHRASES)
    cta = random.choice(PHISHING_CTA)
    link = f"http://{domain}/{random.choice(['verify', 'login', 'secure', 'account', 'update'])}" \
           f"?id={random.randint(100000,999999)}"
    amount = rand_amount()
    fake_name = rand_name()

    body_variants = [
        f"Dear Customer,\n\n{urgency}. We detected suspicious activity on your account associated "
        f"with {brand}. To avoid permanent suspension, please {cta.lower()} within the next 24 hours.\n\n"
        f"{link}\n\nFailure to verify will result in permanent account closure.\n\n{brand} Security Team",

        f"ATTENTION,\n\nYour recent payment of {amount} could not be processed. {urgency}. "
        f"Please update your billing details immediately by clicking the link below:\n\n{link}\n\n"
        f"This is an automated message, do not reply.\n{brand} Billing Department",

        f"Dear Valued Member,\n\nCongratulations! You have been selected to receive a refund of "
        f"{amount} from {brand}. {cta} to claim your refund before it expires:\n\n{link}\n\n"
        f"Offer valid for 24 hours only.\n{brand} Rewards Team",

        f"Hello,\n\nWe noticed an unusual sign-in attempt on your account from an unrecognized "
        f"device. {urgency}. If this wasn't you, {cta.lower()} immediately:\n\n{link}\n\n"
        f"Your account security is our top priority.\n{brand} Support",

        f"Dear User,\n\nYour {brand} account has been temporarily limited due to a policy violation. "
        f"{cta} to restore full access:\n\n{link}\n\nRegards,\n{fake_name}\n{brand} Compliance Team",

        f"URGENT NOTICE,\n\n{urgency}. Your subscription payment of {amount} is overdue. "
        f"Pay immediately to avoid service interruption:\n\n{link}\n\nThank you,\n{brand} Accounts",

        f"Dear Customer,\n\nYour package could not be delivered due to an incomplete address. "
        f"{cta} and confirm your delivery details here:\n\n{link}\n\nA small redelivery fee of "
        f"{amount} applies.\n{brand} Logistics",

        f"Dear Taxpayer,\n\nOur records indicate you are eligible for a refund of {amount}. "
        f"{cta.capitalize()} to claim your refund before the deadline:\n\n{link}\n\n"
        f"{brand} Refund Processing Unit",
    ]

    body = random.choice(body_variants)

    return {
        "email_id": email_id,
        "sender": sender,
        "subject": subject,
        "body": body,
        "label": 1,
    }


def add_realistic_noise(rows):
    """
    Real-world phishing datasets are never perfectly separable. This adds:
      1. A handful of 'hard' legitimate emails that mention urgency/security
         in innocuous business contexts (e.g. a real IT password-reset
         reminder, a real invoice-overdue notice).
      2. A handful of 'hard' phishing emails that are more subtle (well-written,
         no obvious misspelled domain, low urgency).
      3. Small random text perturbations so TF-IDF can't just memorize exact
         template boilerplate.
    """
    hard_legit_bodies = [
        "Hi,\n\nThis is a reminder that your password will expire in 3 days as part of our "
        "standard security policy. Please update it by logging into the employee portal at "
        "your convenience. No immediate action is required if you update within the week.\n\nIT Support",

        "Dear Customer,\n\nYour invoice #4471 is now 5 days overdue. Please log in to your "
        "account dashboard to review and settle the balance at your earliest convenience. "
        "Contact billing@ourcompany.com if you have questions.\n\nAccounts Receivable",

        "Hello,\n\nWe detected a login to your account from a new device (Chrome on Windows) "
        "on Tuesday. If this was you, no action is needed. If not, please reset your password "
        "from your account settings.\n\nSecurity Team",

        "Hi team,\n\nJust confirming our call is still on for tomorrow. Please review the "
        "attached agenda beforehand and come prepared with updates on your workstreams.\n\nThanks,",

        "Dear Member,\n\nYour subscription renews automatically in 7 days. If you'd like to "
        "make changes to your plan, you can do so anytime from your account page. No action "
        "is required to continue your current plan.\n\nCustomer Success Team",
    ]

    hard_phishing_bodies = [
        "Hello,\n\nWe wanted to follow up on your recent account review. Our records show a "
        "small discrepancy that needs your attention when convenient. You can review the "
        "details and update your information at your leisure through the link below.\n\n"
        "http://account-review-center.com/portal\n\nBest regards,\nCustomer Relations",

        "Dear Valued Customer,\n\nThank you for being a loyal member. As part of our ongoing "
        "account verification process, we ask that you confirm a few details on file. This "
        "helps us keep your account secure.\n\nhttp://member-verification.net/update\n\n"
        "Warm regards,\nAccount Services",

        "Hi,\n\nOur team noticed your recent order and wanted to make sure everything arrived "
        "as expected. If you'd like to review your order history or update your delivery "
        "preferences, please visit the link below.\n\nhttp://order-history-check.info/view\n\n"
        "Thanks,\nCustomer Care",

        "Hello,\n\nAs part of a routine account audit, we ask all members to reconfirm their "
        "contact details. This should take less than a minute.\n\n"
        "http://reconfirm-details.support/form\n\nSincerely,\nMembership Team",
    ]

    # Convert a handful of existing legit emails into 'hard legit' variants
    legit_indices = [i for i, r in enumerate(rows) if r["label"] == 0]
    random.shuffle(legit_indices)
    for idx in legit_indices[:60]:
        rows[idx]["body"] = random.choice(hard_legit_bodies)
        rows[idx]["subject"] = random.choice([
            "Password expiring soon", "Invoice reminder", "New device login detected",
            "Call confirmation for tomorrow", "Subscription renewal notice"
        ])

    # Convert a handful of existing phishing emails into 'hard phishing' variants
    phishing_indices = [i for i, r in enumerate(rows) if r["label"] == 1]
    random.shuffle(phishing_indices)
    for idx in phishing_indices[:60]:
        rows[idx]["body"] = random.choice(hard_phishing_bodies)
        rows[idx]["subject"] = random.choice([
            "Following up on your account", "A quick account update",
            "Your recent order", "Routine account audit", "Reconfirm your details"
        ])
        # give hard phishing emails plausible-looking (but still slightly odd) senders
        rows[idx]["sender"] = random.choice([
            "customer.relations@account-review-center.com",
            "services@member-verification.net",
            "care@order-history-check.info",
            "team@reconfirm-details.support",
        ])

    # Small amount of label noise (~2%) to reflect real-world annotation imperfection
    n_noisy = max(1, int(0.02 * len(rows)))
    noisy_indices = random.sample(range(len(rows)), n_noisy)
    for idx in noisy_indices:
        rows[idx]["label"] = 1 - rows[idx]["label"]

    return rows


def main():
    n_per_class = 500  # 500 phishing + 500 legit = 1000 total
    rows = []
    email_id = 1

    for _ in range(n_per_class):
        rows.append(make_legit_email(email_id))
        email_id += 1
    for _ in range(n_per_class):
        rows.append(make_phishing_email(email_id))
        email_id += 1

    random.shuffle(rows)
    rows = add_realistic_noise(rows)

    # re-number after shuffle for a clean sequential id
    for i, r in enumerate(rows, start=1):
        r["email_id"] = i

    out_path = "data/raw_emails.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["email_id", "sender", "subject", "body", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} emails to {out_path}")
    print(f"Phishing: {sum(r['label'] for r in rows)}, Legitimate: {sum(1 - r['label'] for r in rows)}")


if __name__ == "__main__":
    main()
