import unittest
from unittest.mock import patch, MagicMock

# Adjust import path as needed
from analysis.risk_assessor import RiskAssessor, RiskAssessmentResult

class TestRiskAssessor(unittest.TestCase):

    def setUp(self):
        self.assessor = RiskAssessor()

    def test_parse_version_valid(self):
        \"\"\"Test parsing various valid version strings.\"\"\"
        self.assertEqual(str(self.assessor._parse_version("1.2.3")), "1.2.3")
        self.assertEqual(str(self.assessor._parse_version("v2.0.0")), "2.0.0")
        self.assertEqual(str(self.assessor._parse_version("0.1.10-beta.1")), "0.1.10b1")
        self.assertEqual(str(self.assessor._parse_version("=3.14.159")), "3.14.159")
        self.assertEqual(str(self.assessor._parse_version(" 1.0 ")), "1.0") # Handles whitespace
        
    def test_parse_version_invalid(self):
        \"\"\"Test parsing invalid version strings returns None.\"\"\"
        self.assertIsNone(self.assessor._parse_version("invalid-version"))
        self.assertIsNone(self.assessor._parse_version("1.a.b"))
        self.assertIsNone(self.assessor._parse_version(""))
        self.assertIsNone(self.assessor._parse_version(None))

    def test_assess_dependency_update_patch(self):
        \"\"\"Test risk assessment for a patch version update.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.2.3", "1.2.4")
        self.assertEqual(result['risk_level'], 'low')
        self.assertEqual(result['score'], RiskAssessor.PATCH_VERSION_BUMP_RISK)
        self.assertEqual(result['factors']['version_jump'], 'patch')
        self.assertIn("Patch version bump detected", result['explanation'])

    def test_assess_dependency_update_minor(self):
        \"\"\"Test risk assessment for a minor version update.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.2.3", "1.3.0")
        self.assertEqual(result['risk_level'], 'medium')
        self.assertEqual(result['score'], RiskAssessor.MINOR_VERSION_BUMP_RISK)
        self.assertEqual(result['factors']['version_jump'], 'minor')
        self.assertIn("Minor version bump detected", result['explanation'])

    def test_assess_dependency_update_major(self):
        \"\"\"Test risk assessment for a major version update.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.2.3", "2.0.0")
        self.assertEqual(result['risk_level'], 'high') # Default threshold makes 0.7 high
        self.assertEqual(result['score'], RiskAssessor.MAJOR_VERSION_BUMP_RISK)
        self.assertEqual(result['factors']['version_jump'], 'major')
        self.assertIn("Major version bump detected", result['explanation'])
        
    def test_assess_dependency_update_major_critical_threshold(self):
         \"\"\"Test risk assessment for a major version update hitting critical threshold.\"\"\"
         # Temporarily increase major bump risk for this test
         original_risk = RiskAssessor.MAJOR_VERSION_BUMP_RISK
         RiskAssessor.MAJOR_VERSION_BUMP_RISK = 0.85
         result = self.assessor.assess_dependency_update("critical-pkg", "1.9.9", "2.0.0")
         self.assertEqual(result['risk_level'], 'critical') 
         self.assertEqual(result['score'], 0.85)
         RiskAssessor.MAJOR_VERSION_BUMP_RISK = original_risk # Restore original value

    def test_assess_dependency_update_no_change(self):
        \"\"\"Test risk assessment when versions are identical.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.2.3", "1.2.3")
        self.assertEqual(result['risk_level'], 'low') # Score is 0.0
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['factors']['version_jump'], 'none')
        self.assertIn("No version change", result['explanation'])
        
    def test_assess_dependency_update_downgrade(self):
        \"\"\"Test risk assessment for a version downgrade.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.3.0", "1.2.4")
        self.assertEqual(result['risk_level'], 'low') # Score is 0.0
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['factors']['version_jump'], 'none') # Currently classifies as 'none'
        self.assertIn("downgrade detected", result['explanation'])

    def test_assess_dependency_update_invalid_current(self):
        \"\"\"Test risk assessment when current version is invalid.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "invalid", "1.0.0")
        self.assertEqual(result['risk_level'], 'medium') # Defaults to medium risk
        self.assertEqual(result['score'], RiskAssessor.MINOR_VERSION_BUMP_RISK)
        self.assertEqual(result['factors']['version_jump'], 'unknown_current')
        self.assertIn("current version unknown/invalid", result['explanation'])

    def test_assess_dependency_update_invalid_new(self):
        \"\"\"Test risk assessment when new version is invalid.\"\"\"
        result = self.assessor.assess_dependency_update("pkg", "1.0.0", "bad-version")
        self.assertEqual(result['risk_level'], 'medium') # Defaults to medium risk
        self.assertEqual(result['score'], RiskAssessor.MINOR_VERSION_BUMP_RISK)
        self.assertEqual(result['factors']['version_jump'], 'parse_error')
        self.assertIn("Could not parse versions", result['explanation'])
        
    def test_assess_dependency_update_both_invalid(self):
         \"\"\"Test risk assessment when both versions are invalid.\"\"\"
         result = self.assessor.assess_dependency_update("pkg", "-", "~")
         self.assertEqual(result['risk_level'], 'medium') # Defaults to medium risk
         self.assertEqual(result['score'], RiskAssessor.MINOR_VERSION_BUMP_RISK)
         self.assertEqual(result['factors']['version_jump'], 'parse_error')
         self.assertIn("Could not parse versions", result['explanation'])

    def test_assess_code_change_placeholder(self):
        \"\"\"Test the placeholder implementation for code change assessment.\"\"\"
        # This test just ensures the placeholder runs without error
        # and returns the expected placeholder structure.
        # TODO: Update this test when CodeDiffer is integrated.
        result = self.assessor.assess_code_change("before", "after", "python")
        self.assertEqual(result['risk_level'], 'medium')
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['factors'], {'diff_analysis': 'not_implemented'})
        self.assertIn("pending implementation", result['explanation'])

if __name__ == '__main__':
    unittest.main() 