import unittest
from exchange_session_audit_v279 import sessions


class SessionTests(unittest.TestCase):
    def test_annual_holiday_successors(self):
        cal=sessions();nxt=dict(zip(cal[:-1],cal[1:]))
        self.assertEqual(nxt['2024-02-08'],'2024-02-19')
        self.assertEqual(nxt['2025-01-27'],'2025-02-05')
        self.assertEqual(nxt['2026-09-24'],'2026-09-28')
        self.assertEqual(nxt['2026-09-30'],'2026-10-08')

    def test_makeup_workdays_do_not_open_exchange(self):
        cal=sessions()
        self.assertNotIn('2024-02-18',cal)
        self.assertNotIn('2025-02-08',cal)
        self.assertNotIn('2026-05-09',cal)

    def test_outside_verified_scope_rejected(self):
        with self.assertRaises(ValueError):sessions('2023-01-01','2023-12-31')
        with self.assertRaises(ValueError):sessions('2026-10-01','2027-01-01')


if __name__=='__main__':unittest.main()
